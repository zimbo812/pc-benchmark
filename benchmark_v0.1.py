"""
PC Benchmark — програма визначення продуктивності ПК
з використанням алгоритмів стрес-тестування.

Модулі тестування:
  1. CPU (одне ядро)   — набір обчислювальних алгоритмів
  2. CPU (усі ядра)    — ті самі алгоритми паралельно у N процесах
  3. Оперативна пам'ять — пропускна здатність запису / читання / копіювання
  4. Накопичувач        — послідовний запис/читання, випадкові 4K операції
  5. Стрес-тест         — тривале повне навантаження всіх ядер з перевіркою
                          правильності обчислень і виявленням тротлінгу

Запуск:
  python benchmark.py                  — графічний інтерфейс
  python benchmark.py --cli            — консольний режим
  python benchmark.py --cli --stress 60 --no-disk
Залежності: лише стандартна бібліотека Python 3.8+ (psutil — необов'язково).
"""

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import random
import statistics
import sys
import tempfile
import threading
import time
import zlib

try:
    import psutil  # необов'язково: частота CPU, завантаження
except ImportError:
    psutil = None

APP_NAME = "PC Benchmark"
VERSION = "0.1"

# ---------------------------------------------------------------------------
#  Алгоритми навантаження (функції верхнього рівня — потрібні для multiprocessing)
#  Кожна функція повертає контрольне значення для перевірки правильності.
# ---------------------------------------------------------------------------


def wl_primes(n=3_000_000):
    """Решето Ератосфена: кількість простих чисел до n (цілочисельні операції, пам'ять)."""
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytes(len(range(i * i, n + 1, i)))
    return sum(sieve)


def wl_mandelbrot(w=200, h=150, max_iter=200):
    """Множина Мандельброта: операції з плаваючою комою, розгалуження."""
    total = 0
    for py in range(h):
        y0 = py / h * 2.4 - 1.2
        for px in range(w):
            x0 = px / w * 3.5 - 2.5
            x = y = 0.0
            i = 0
            while x * x + y * y <= 4.0 and i < max_iter:
                x, y = x * x - y * y + x0, 2.0 * x * y + y0
                i += 1
            total += i
    return total


def wl_matrix(n=110, seed=1):
    """Множення матриць n×n (FPU, кеш)."""
    rnd = random.Random(seed)
    a = [[rnd.random() for _ in range(n)] for _ in range(n)]
    b = [[rnd.random() for _ in range(n)] for _ in range(n)]
    bt = list(zip(*b))
    c = [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]
    return round(sum(map(sum, c)), 6)


def wl_sha256(mb=96):
    """Хешування SHA-256 (криптографічні операції, вбудовані в CPU розширення)."""
    block = bytes(range(256)) * 4096  # 1 МіБ
    h = hashlib.sha256()
    for _ in range(mb):
        h.update(block)
    return h.hexdigest()


def wl_zlib(mb=6, seed=7):
    """Стиснення/розпакування DEFLATE (алгоритм LZ77 + Хаффман)."""
    rnd = random.Random(seed)
    words = [b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta", b"eta",
             b"theta", b"iota", b"kappa", b"lambda", b"mu", b"benchmark", b"data"]
    chunk = b" ".join(rnd.choices(words, k=20000))
    data = (chunk * (mb * 1024 * 1024 // len(chunk) + 1))[: mb * 1024 * 1024]
    comp = zlib.compress(data, 6)
    if zlib.decompress(comp) != data:
        raise RuntimeError("zlib: дані після розпакування не збігаються")
    return zlib.crc32(comp)


def wl_sort(n=600_000, seed=3):
    """Генерація та сортування масиву (Timsort), робота з об'єктами в пам'яті."""
    rnd = random.Random(seed)
    arr = [rnd.random() for _ in range(n)]
    arr.sort()
    return arr[n // 2]


# id: (назва, функція, параметри, еталонний час у секундах = 1000 балів)
CPU_TESTS = {
    "primes":     ("Решето Ератосфена",     wl_primes,     {"n": 3_000_000}, 0.040),
    "mandelbrot": ("Множина Мандельброта",  wl_mandelbrot, {},               0.280),
    "matrix":     ("Множення матриць",      wl_matrix,     {"n": 110},       0.155),
    "sha256":     ("Хешування SHA-256",     wl_sha256,     {"mb": 96},       0.130),
    "zlib":       ("Стиснення zlib",        wl_zlib,       {"mb": 6},        0.350),
    "sort":       ("Сортування масиву",     wl_sort,       {"n": 600_000},   0.400),
}

# Полегшені параметри для композитного завдання (багатоядерний і стрес-тест)
COMPOSITE_PARAMS = {
    "primes": {"n": 1_000_000},
    "mandelbrot": {"w": 100, "h": 75},
    "matrix": {"n": 60},
    "sha256": {"mb": 24},
    "zlib": {"mb": 2},
    "sort": {"n": 150_000},
}


def wl_composite(_=None):
    """Одне композитне завдання: усі алгоритми з полегшеними параметрами."""
    return tuple(CPU_TESTS[k][1](**p) for k, p in COMPOSITE_PARAMS.items())


def timed_composite(_=None):
    t = time.perf_counter()
    r = wl_composite()
    return time.perf_counter() - t, r


# Еталонні значення для інших підсистем (1000 балів)
REF_COMPOSITE_SEC = 0.35        # час одного композитного завдання на еталоні (1 потік)
REF_MULTI_RATE = 12.0           # пропускна здатність еталону на всіх ядрах, завд/с
REF_MEM_GBPS = {"write": 8.0, "read": 12.0, "copy": 6.0}
REF_DISK = {"seq_write": 400.0, "seq_read": 1500.0, "rnd_write_iops": 1500.0, "rnd_read_iops": 30000.0}
WEIGHTS = {"cpu_single": 0.30, "cpu_multi": 0.30, "memory": 0.20, "disk": 0.20}


def geomean(values):
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))



# ---------------------------------------------------------------------------
#  Інформація про систему
# ---------------------------------------------------------------------------


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or "Невідомо"


def total_ram_gb():
    try:
        if psutil:
            return psutil.virtual_memory().total / 1024 ** 3
        if sys.platform == "win32":
            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1024 ** 3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except Exception:
        return 0.0


def system_info():
    return {
        "Процесор": cpu_name(),
        "Логічних ядер": os.cpu_count(),
        "ОЗП, ГБ": round(total_ram_gb(), 1),
        "ОС": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
#  Ядро бенчмарку (не залежить від інтерфейсу)
# ---------------------------------------------------------------------------


class Stopped(Exception):
    pass


class Benchmark:
    """
    Виконує вибрані тести. Події передаються через emit(kind, data):
      log(str), progress(float 0..100), result(dict), stress_point((t, jobs_per_s)),
      category(dict), done(dict звіт)
    """

    def __init__(self, cfg, emit, stop_event):
        self.cfg = cfg
        self.emit = emit
        self.stop = stop_event
        self.results = []
        self.categories = {}
        self.stress = None
        steps = 0
        if cfg["cpu_single"]:
            steps += len(CPU_TESTS)
        if cfg["cpu_multi"]:
            steps += 2
        if cfg["memory"]:
            steps += 3
        if cfg["disk"]:
            steps += 4
        if cfg["stress"]:
            steps += 10
        self.total_steps = max(steps, 1)
        self.done_steps = 0

    # --- допоміжне ---
    def _check(self):
        if self.stop.is_set():
            raise Stopped()

    def _step(self, n=1):
        self.done_steps += n
        self.emit("progress", min(100.0, 100.0 * self.done_steps / self.total_steps))

    def _add(self, category, test, value, unit, score):
        r = {"category": category, "test": test, "value": round(value, 3),
             "unit": unit, "score": round(score)}
        self.results.append(r)
        self.emit("result", r)

    def _category(self, key, name, score):
        self.categories[key] = {"name": name, "score": round(score)}
        self.emit("category", {"key": key, "name": name, "score": round(score)})

    # --- головний метод ---
    def run(self):
        started = time.time()
        status = "Завершено"
        try:
            self.emit("log", "Старт тестування…")
            if self.cfg["cpu_single"]:
                self.cpu_single()
            if self.cfg["cpu_multi"]:
                self.cpu_multi()
            if self.cfg["memory"]:
                self.memory()
            if self.cfg["disk"]:
                self.disk()
            if self.cfg["stress"]:
                self.stress_test()
        except Stopped:
            status = "Перервано користувачем"
            self.emit("log", "Тестування перервано.")
        except Exception as e:  # noqa: BLE001
            status = f"Помилка: {e}"
            self.emit("log", f"ПОМИЛКА: {e!r}")

        total = 0.0
        if self.categories:
            wsum = sum(WEIGHTS[k] for k in self.categories if k in WEIGHTS)
            if wsum:
                total = math.exp(sum(WEIGHTS[k] * math.log(max(v["score"], 1))
                                     for k, v in self.categories.items() if k in WEIGHTS) / wsum)
        report = {
            "program": f"{APP_NAME} {VERSION}",
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "duration_sec": round(time.time() - started, 1),
            "status": status,
            "system": system_info(),
            "config": {k: v for k, v in self.cfg.items() if k != "disk_dir"},
            "results": self.results,
            "categories": self.categories,
            "stress": self.stress,
            "total_score": round(total),
        }
        self.emit("progress", 100.0)
        self.emit("done", report)
        return report

    # --- 1. CPU, одне ядро ---
    def cpu_single(self):
        self.emit("log", "== CPU: одне ядро ==")
        repeats = 3
        scores = []
        for key, (name, fn, params, ref) in CPU_TESTS.items():
            self._check()
            times, ref_result = [], None
            for _ in range(repeats):
                t0 = time.perf_counter()
                r = fn(**params)
                times.append(time.perf_counter() - t0)
                if ref_result is None:
                    ref_result = r
                elif r != ref_result:
                    raise RuntimeError(f"{name}: результати повторів не збігаються (нестабільність)")
            best = min(times)
            score = 1000.0 * ref / best
            scores.append(score)
            self._add("CPU (1 ядро)", name, best * 1000, "мс", score)
            self.emit("log", f"  {name}: {best * 1000:.1f} мс (σ={statistics.pstdev(times) * 1000:.1f}) → {score:.0f}")
            self._step()
        self._category("cpu_single", "CPU (1 ядро)", geomean(scores))

    # --- 2. CPU, усі ядра ---
    def cpu_multi(self):
        n = self.cfg["threads"]
        self.emit("log", f"== CPU: {n} процес(ів) ==")
        ctx = mp.get_context("spawn")
        with ctx.Pool(n) as pool:
            pool.map(timed_composite, range(n))          # прогрів (запуск процесів)
            self._check()
            t1 = min(pool.apply(timed_composite)[0] for _ in range(3))   # одне завдання без конкуренції
            self._step()
            self._check()
            jobs = n * 3
            t0 = time.perf_counter()
            res = pool.map(timed_composite, range(jobs), chunksize=1)
            tn = time.perf_counter() - t0
            golden = res[0][1]
            if any(r[1] != golden for r in res):
                raise RuntimeError("розбіжність результатів між процесами (нестабільність CPU/ОЗП)")
            self._step()
        throughput = jobs / tn
        speedup = throughput * t1
        efficiency = speedup / n * 100
        score = 1000.0 * throughput / REF_MULTI_RATE
        self._add("CPU (усі ядра)", "Композитне завдання, 1 потік", t1 * 1000, "мс", 1000.0 * REF_COMPOSITE_SEC / t1)
        self._add("CPU (усі ядра)", f"Пропускна здатність, {n} потоків", throughput, "завд/с", score)
        self._add("CPU (усі ядра)", "Прискорення (закон Амдала)", speedup, "×", 0)
        self._add("CPU (усі ядра)", "Ефективність масштабування", efficiency, "%", 0)
        self.emit("log", f"  1 потік: {t1:.3f} с; {n} потоків: {throughput:.2f} завд/с; "
                         f"прискорення ×{speedup:.2f} ({efficiency:.0f}%)")
        self._category("cpu_multi", f"CPU ({n} потоків)", score)

    # --- 3. Оперативна пам'ять ---
    def memory(self):
        size_mb = self.cfg.get("mem_mb", 256)
        n = size_mb * 1024 * 1024
        self.emit("log", f"== Оперативна пам'ять (блок {size_mb} МБ) ==")
        src = bytearray(n)
        dst = bytearray(n)
        src_c = (ctypes.c_char * n).from_buffer(src)
        repeats = 5
        scores = []

        def best_of(fn):
            ts = []
            for _ in range(repeats):
                self._check()
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            return min(ts)

        # запис: memset
        t = best_of(lambda: ctypes.memset(src_c, 0x5A, n))
        gbps = n / t / 1e9
        scores.append(1000 * gbps / REF_MEM_GBPS["write"])
        self._add("Пам'ять", "Запис (memset)", gbps, "ГБ/с", scores[-1])
        self._step()
        # читання: послідовне сканування пошуком відсутнього байта (memchr)
        t = best_of(lambda: src.find(b"\x00"))
        gbps = n / t / 1e9
        scores.append(1000 * gbps / REF_MEM_GBPS["read"])
        self._add("Пам'ять", "Читання (сканування)", gbps, "ГБ/с", scores[-1])
        self._step()
        # копіювання: memmove
        t = best_of(lambda: dst.__setitem__(slice(None), src))
        gbps = n / t / 1e9
        scores.append(1000 * gbps / REF_MEM_GBPS["copy"])
        self._add("Пам'ять", "Копіювання (memmove)", gbps, "ГБ/с", scores[-1])
        self._step()
        if dst[n - 1] != 0x5A:
            raise RuntimeError("помилка перевірки даних у пам'яті")
        del src_c
        for r in self.results[-3:]:
            self.emit("log", f"  {r['test']}: {r['value']:.2f} ГБ/с → {r['score']}")
        self._category("memory", "Пам'ять", geomean(scores))

    # --- 4. Накопичувач ---
    def disk(self):
        size_mb = self.cfg["disk_mb"]
        block = 4 * 1024 * 1024
        directory = self.cfg.get("disk_dir") or tempfile.gettempdir()
        path = os.path.join(directory, f"pcbench_{os.getpid()}.tmp")
        self.emit("log", f"== Накопичувач ({size_mb} МБ, {directory}) ==")
        data = os.urandom(block)          # випадкові дані — без вигоди від стиснення
        blocks = size_mb * 1024 * 1024 // block
        scores = []
        try:
            # послідовний запис
            t0 = time.perf_counter()
            with open(path, "wb", buffering=0) as f:
                for _ in range(blocks):
                    self._check()
                    f.write(data)
                os.fsync(f.fileno())
            mbps = blocks * block / (time.perf_counter() - t0) / 1e6
            scores.append(1000 * mbps / REF_DISK["seq_write"])
            self._add("Накопичувач", "Послідовний запис 4 МБ", mbps, "МБ/с", scores[-1])
            self._step()

            # послідовне читання
            t0 = time.perf_counter()
            with open(path, "rb", buffering=0) as f:
                while f.read(block):
                    self._check()
            mbps = blocks * block / (time.perf_counter() - t0) / 1e6
            scores.append(1000 * mbps / REF_DISK["seq_read"])
            self._add("Накопичувач", "Послідовне читання 4 МБ*", mbps, "МБ/с", scores[-1])
            self._step()

            # випадковий запис 4K з fsync після кожної операції
            rnd = random.Random(42)
            page = os.urandom(4096)
            pages = blocks * block // 4096
            ops, deadline = 0, time.perf_counter() + 3.0
            t0 = time.perf_counter()
            with open(path, "r+b", buffering=0) as f:
                while time.perf_counter() < deadline and ops < 5000:
                    f.seek(rnd.randrange(pages) * 4096)
                    f.write(page)
                    os.fsync(f.fileno())
                    ops += 1
                    if ops % 50 == 0:
                        self._check()
            iops = ops / (time.perf_counter() - t0)
            scores.append(1000 * iops / REF_DISK["rnd_write_iops"])
            self._add("Накопичувач", "Випадковий запис 4K (sync)", iops, "IOPS", scores[-1])
            self._step()

            # випадкове читання 4K
            ops, deadline = 0, time.perf_counter() + 2.0
            t0 = time.perf_counter()
            with open(path, "rb", buffering=0) as f:
                while time.perf_counter() < deadline:
                    f.seek(rnd.randrange(pages) * 4096)
                    f.read(4096)
                    ops += 1
                    if ops % 500 == 0:
                        self._check()
            iops = ops / (time.perf_counter() - t0)
            scores.append(1000 * iops / REF_DISK["rnd_read_iops"])
            self._add("Накопичувач", "Випадкове читання 4K*", iops, "IOPS", scores[-1])
            self._step()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        for r in self.results[-4:]:
            self.emit("log", f"  {r['test']}: {r['value']:.1f} {r['unit']} → {r['score']}")
        self.emit("log", "  * читання може прискорюватися кешем файлової системи ОС")
        self._category("disk", "Накопичувач", geomean(scores))

    # --- 5. Стрес-тест ---
    def stress_test(self):
        n = self.cfg["threads"]
        duration = self.cfg["stress_seconds"]
        self.emit("log", f"== Стрес-тест: {n} процес(ів), {duration} с ==")
        golden = wl_composite()          # еталонний результат для перевірки правильності
        ctx = mp.get_context("spawn")
        points, errors, jobs_total = [], 0, 0
        freqs = []
        last_progress = 0
        with ctx.Pool(n) as pool:
            pool.map(wl_composite, range(n))  # прогрів
            start = time.perf_counter()
            while True:
                elapsed = time.perf_counter() - start
                if elapsed >= duration or self.stop.is_set():
                    break
                t0 = time.perf_counter()
                res = pool.map(wl_composite, range(n), chunksize=1)
                dt = time.perf_counter() - t0
                bad = sum(1 for r in res if r != golden)
                errors += bad
                jobs_total += n
                if bad:
                    self.emit("log", f"  ! {bad} помилкових результатів на {elapsed:.0f} с")
                rate = n / dt
                t_rel = time.perf_counter() - start
                points.append((round(t_rel, 2), round(rate, 3)))
                self.emit("stress_point", (t_rel, rate))
                if psutil:
                    try:
                        f = psutil.cpu_freq()
                        if f:
                            freqs.append(f.current)
                    except Exception:
                        pass
                pr = int(10 * min(t_rel / duration, 1.0))
                if pr > last_progress:
                    self._step(pr - last_progress)
                    last_progress = pr
        if last_progress < 10:
            self._step(10 - last_progress)

        rates = [p[1] for p in points]
        k = max(1, len(rates) // 5)
        first = statistics.mean(rates[:k]) if rates else 0
        last = statistics.mean(rates[-k:]) if rates else 0
        degradation = (first - last) / first * 100 if first else 0
        cv = statistics.pstdev(rates) / statistics.mean(rates) * 100 if len(rates) > 1 else 0
        if errors:
            verdict = "НЕСТАБІЛЬНО: виявлено помилки обчислень"
        elif degradation > 10:
            verdict = "Стабільно, але ймовірний тротлінг (падіння продуктивності)"
        elif not points:
            verdict = "Недостатньо даних"
        else:
            verdict = "Стабільно"
        self.stress = {
            "duration_sec": round(points[-1][0] if points else 0, 1),
            "threads": n, "jobs": jobs_total, "errors": errors,
            "avg_rate": round(statistics.mean(rates), 3) if rates else 0,
            "min_rate": round(min(rates), 3) if rates else 0,
            "max_rate": round(max(rates), 3) if rates else 0,
            "degradation_pct": round(degradation, 1),
            "variation_pct": round(cv, 1),
            "avg_freq_mhz": round(statistics.mean(freqs)) if freqs else None,
            "verdict": verdict,
            "points": points,
        }
        avg = self.stress["avg_rate"]
        self._add("Стрес-тест", "Середня пропускна здатність", avg, "завд/с", 1000 * avg / REF_MULTI_RATE)
        self._add("Стрес-тест", "Падіння продуктивності", degradation, "%", 0)
        self._add("Стрес-тест", "Коефіцієнт варіації", cv, "%", 0)
        self._add("Стрес-тест", "Помилки обчислень", errors, "шт", 0)
        self.emit("log", f"  Завдань: {jobs_total}, помилок: {errors}, падіння: {degradation:.1f}%, "
                         f"варіація: {cv:.1f}% → {verdict}")
        self.emit("stress_done", self.stress)


def report_to_text(rep):
    lines = [f"{rep['program']} — звіт про тестування", "=" * 60,
             f"Дата: {rep['date']}   Тривалість: {rep['duration_sec']} с   Статус: {rep['status']}", "",
             "СИСТЕМА:"]
    lines += [f"  {k}: {v}" for k, v in rep["system"].items()]
    lines += ["", "РЕЗУЛЬТАТИ:", f"  {'Категорія':<16}{'Тест':<36}{'Значення':>14}  {'Од.':<7}{'Бали':>6}"]
    for r in rep["results"]:
        sc = r["score"] if r["score"] else ""
        lines.append(f"  {r['category']:<16}{r['test']:<36}{r['value']:>14.2f}  {r['unit']:<7}{sc:>6}")
    lines += ["", "БАЛИ ЗА КАТЕГОРІЯМИ:"]
    lines += [f"  {c['name']:<24}{c['score']:>8}" for c in rep["categories"].values()]
    if rep.get("stress"):
        s = rep["stress"]
        lines += ["", "СТРЕС-ТЕСТ:", f"  Тривалість: {s['duration_sec']} с, потоків: {s['threads']}",
                  f"  Завдань: {s['jobs']}, помилок: {s['errors']}",
                  f"  Пропускна здатність: сер. {s['avg_rate']}, мін. {s['min_rate']}, макс. {s['max_rate']} завд/с",
                  f"  Падіння продуктивності: {s['degradation_pct']}%, варіація: {s['variation_pct']}%",
                  f"  Висновок: {s['verdict']}"]
    lines += ["", f"ЗАГАЛЬНИЙ БАЛ: {rep['total_score']}",
              "(1000 балів = умовний еталонний ПК; більше — краще)"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Консольний режим
# ---------------------------------------------------------------------------


def run_cli(args):
    cfg = {
        "cpu_single": not args.no_cpu, "cpu_multi": not args.no_cpu,
        "memory": not args.no_mem, "disk": not args.no_disk,
        "stress": args.stress > 0, "stress_seconds": args.stress,
        "threads": args.threads or os.cpu_count(), "disk_mb": args.disk_mb,
        "mem_mb": 256, "disk_dir": args.disk_dir,
    }
    print(f"{APP_NAME} {VERSION}")
    for k, v in system_info().items():
        print(f"  {k}: {v}")
    report = {}

    def emit(kind, data):
        if kind == "log":
            print(data, flush=True)
        elif kind == "done":
            report.update(data)

    stop = threading.Event()
    try:
        Benchmark(cfg, emit, stop).run()
    except KeyboardInterrupt:
        stop.set()
    print()
    print(report_to_text(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nЗвіт збережено: {args.out}")


# ---------------------------------------------------------------------------
#  Графічний інтерфейс (tkinter)
# ---------------------------------------------------------------------------


def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(f"{APP_NAME} {VERSION} — визначення продуктивності ПК")
            self.geometry("980x700")
            self.minsize(860, 600)
            self.q = queue.Queue()
            self.stop_event = threading.Event()
            self.worker = None
            self.report = None
            self.points = []
            self._build()
            self.after(100, self._poll)

        # ---------- побудова вікна ----------
        def _build(self):
            style = ttk.Style(self)
            if "vista" in style.theme_names():
                style.theme_use("vista")
            style.configure("Score.TLabel", font=("Segoe UI", 20, "bold"))
            style.configure("Head.TLabel", font=("Segoe UI", 10, "bold"))

            top = ttk.Frame(self, padding=8)
            top.pack(fill="x")
            info = system_info()
            ttk.Label(top, text="Система", style="Head.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(top, text="   ".join(f"{k}: {v}" for k, v in info.items()),
                      wraplength=900).grid(row=1, column=0, sticky="w")

            opts = ttk.LabelFrame(self, text="Параметри тестування", padding=8)
            opts.pack(fill="x", padx=8)
            self.v = {k: tk.BooleanVar(value=True) for k in ("cpu_single", "cpu_multi", "memory", "disk", "stress")}
            labels = {"cpu_single": "CPU (1 ядро)", "cpu_multi": "CPU (усі ядра)", "memory": "Оперативна пам'ять",
                      "disk": "Накопичувач", "stress": "Стрес-тест"}
            for i, (k, t) in enumerate(labels.items()):
                ttk.Checkbutton(opts, text=t, variable=self.v[k]).grid(row=0, column=i, padx=6, sticky="w")

            self.v_threads = tk.IntVar(value=os.cpu_count())
            self.v_stress = tk.IntVar(value=60)
            self.v_disk = tk.IntVar(value=512)
            self.v_dir = tk.StringVar(value=tempfile.gettempdir())
            ttk.Label(opts, text="Потоків:").grid(row=1, column=0, sticky="e", pady=(8, 0))
            ttk.Spinbox(opts, from_=1, to=(os.cpu_count() or 1) * 2, width=6,
                        textvariable=self.v_threads).grid(row=1, column=1, sticky="w", pady=(8, 0))
            ttk.Label(opts, text="Стрес-тест, с:").grid(row=1, column=2, sticky="e", pady=(8, 0))
            ttk.Spinbox(opts, from_=10, to=3600, increment=10, width=6,
                        textvariable=self.v_stress).grid(row=1, column=3, sticky="w", pady=(8, 0))
            ttk.Label(opts, text="Файл диска, МБ:").grid(row=1, column=4, sticky="e", pady=(8, 0))
            ttk.Combobox(opts, values=[128, 256, 512, 1024, 2048], width=6, state="readonly",
                         textvariable=self.v_disk).grid(row=1, column=5, sticky="w", pady=(8, 0))
            ttk.Label(opts, text="Тека для тесту диска:").grid(row=2, column=0, sticky="e", pady=(6, 0))
            ttk.Entry(opts, textvariable=self.v_dir, width=60).grid(row=2, column=1, columnspan=4, sticky="we", pady=(6, 0))
            ttk.Button(opts, text="Огляд…", command=self._choose_dir).grid(row=2, column=5, sticky="w", pady=(6, 0))

            bar = ttk.Frame(self, padding=8)
            bar.pack(fill="x")
            self.btn_start = ttk.Button(bar, text="▶ Запустити", command=self.start)
            self.btn_stop = ttk.Button(bar, text="■ Зупинити", command=self.stop, state="disabled")
            self.btn_save = ttk.Button(bar, text="💾 Зберегти звіт", command=self.save, state="disabled")
            self.btn_start.pack(side="left")
            self.btn_stop.pack(side="left", padx=6)
            self.btn_save.pack(side="left")
            self.lbl_total = ttk.Label(bar, text="Загальний бал: —", style="Score.TLabel")
            self.lbl_total.pack(side="right")
            self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
            self.progress.pack(fill="x", padx=8)
            self.lbl_status = ttk.Label(self, text="Готово до запуску", padding=(8, 2))
            self.lbl_status.pack(fill="x")

            nb = ttk.Notebook(self)
            nb.pack(fill="both", expand=True, padx=8, pady=8)

            # вкладка результатів
            tab_res = ttk.Frame(nb)
            nb.add(tab_res, text="Результати")
            cols = ("category", "test", "value", "unit", "score")
            self.tree = ttk.Treeview(tab_res, columns=cols, show="headings")
            for c, t, w, a in (("category", "Категорія", 130, "w"), ("test", "Тест", 300, "w"),
                               ("value", "Значення", 120, "e"), ("unit", "Од.", 70, "w"),
                               ("score", "Бали", 80, "e")):
                self.tree.heading(c, text=t)
                self.tree.column(c, width=w, anchor=a)
            self.tree.tag_configure("cat", background="#e8f0fe", font=("Segoe UI", 9, "bold"))
            sb = ttk.Scrollbar(tab_res, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")

            # вкладка стрес-тесту
            tab_st = ttk.Frame(nb)
            nb.add(tab_st, text="Стрес-тест")
            self.canvas = tk.Canvas(tab_st, background="white", highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)
            self.canvas.bind("<Configure>", lambda e: self._draw_chart())
            self.lbl_stress = ttk.Label(tab_st, text="Графік пропускної здатності під тривалим навантаженням",
                                        padding=6)
            self.lbl_stress.pack(fill="x")

            # вкладка журналу
            tab_log = ttk.Frame(nb)
            nb.add(tab_log, text="Журнал")
            self.log = tk.Text(tab_log, font=("Consolas", 9), wrap="word")
            self.log.pack(fill="both", expand=True)
            self.nb = nb

        def _choose_dir(self):
            d = filedialog.askdirectory(initialdir=self.v_dir.get())
            if d:
                self.v_dir.set(d)

        # ---------- керування ----------
        def start(self):
            cfg = {k: v.get() for k, v in self.v.items()}
            if not any(cfg.values()):
                messagebox.showwarning(APP_NAME, "Оберіть хоча б один тест.")
                return
            try:
                cfg.update(threads=max(1, int(self.v_threads.get())), stress_seconds=max(5, int(self.v_stress.get())),
                           disk_mb=int(self.v_disk.get()), mem_mb=256, disk_dir=self.v_dir.get())
            except (tk.TclError, ValueError):
                messagebox.showerror(APP_NAME, "Некоректні параметри.")
                return
            if cfg["disk"] and not os.path.isdir(cfg["disk_dir"]):
                messagebox.showerror(APP_NAME, "Тека для тесту диска не існує.")
                return
            self.tree.delete(*self.tree.get_children())
            self.log.delete("1.0", "end")
            self.points = []
            self._draw_chart()
            self.report = None
            self.lbl_total.config(text="Загальний бал: …")
            self.stop_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_save.config(state="disabled")
            self.btn_stop.config(state="normal")
            bench = Benchmark(cfg, lambda k, d: self.q.put((k, d)), self.stop_event)
            self.worker = threading.Thread(target=bench.run, daemon=True)
            self.worker.start()

        def stop(self):
            self.stop_event.set()
            self.lbl_status.config(text="Зупинка після поточного кроку…")

        def save(self):
            if not self.report:
                return
            path = filedialog.asksaveasfilename(
                defaultextension=".txt", initialfile=f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}",
                filetypes=[("Текстовий звіт", "*.txt"), ("JSON", "*.json")])
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                if path.lower().endswith(".json"):
                    json.dump(self.report, f, ensure_ascii=False, indent=2)
                else:
                    f.write(report_to_text(self.report))
            self.lbl_status.config(text=f"Звіт збережено: {path}")

        # ---------- обробка подій з робочого потоку ----------
        def _poll(self):
            try:
                while True:
                    kind, data = self.q.get_nowait()
                    getattr(self, f"_on_{kind}")(data)
            except queue.Empty:
                pass
            self.after(100, self._poll)

        def _on_log(self, msg):
            self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self.log.see("end")
            if not msg.startswith("  "):
                self.lbl_status.config(text=msg.strip("= "))

        def _on_progress(self, p):
            self.progress["value"] = p

        def _on_result(self, r):
            self.tree.insert("", "end", values=(r["category"], r["test"], f"{r['value']:.2f}", r["unit"],
                                                r["score"] or ""))

        def _on_category(self, c):
            self.tree.insert("", "end", values=("", f"Підсумок: {c['name']}", "", "", c["score"]), tags=("cat",))

        def _on_stress_point(self, p):
            self.points.append(p)
            self._draw_chart()
            if len(self.points) == 1:
                self.nb.select(1)

        def _on_stress_done(self, s):
            self.lbl_stress.config(text=f"Помилок: {s['errors']}   Падіння: {s['degradation_pct']}%   "
                                        f"Варіація: {s['variation_pct']}%   Висновок: {s['verdict']}")

        def _on_done(self, rep):
            self.report = rep
            self.lbl_total.config(text=f"Загальний бал: {rep['total_score']}")
            self.lbl_status.config(text=f"{rep['status']} за {rep['duration_sec']} с")
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.btn_save.config(state="normal")

        # ---------- графік ----------
        def _draw_chart(self):
            c = self.canvas
            c.delete("all")
            w, h = c.winfo_width(), c.winfo_height()
            if w < 50 or h < 50:
                return
            l, r, t, b = 60, 20, 20, 40
            c.create_line(l, t, l, h - b, fill="#555")
            c.create_line(l, h - b, w - r, h - b, fill="#555")
            c.create_text(l, t - 8, text="завд/с", anchor="w", fill="#555")
            c.create_text(w - r, h - b + 25, text="час, с", anchor="e", fill="#555")
            if not self.points:
                c.create_text(w / 2, h / 2, text="Дані з'являться під час стрес-тесту", fill="#999")
                return
            tmax = max(self.points[-1][0], 1)
            vmax = max(p[1] for p in self.points) * 1.15
            for i in range(5):  # сітка
                y = h - b - (h - t - b) * i / 4
                c.create_line(l, y, w - r, y, fill="#eee")
                c.create_text(l - 5, y, text=f"{vmax * i / 4:.1f}", anchor="e", fill="#555")
            for i in range(6):
                x = l + (w - l - r) * i / 5
                c.create_text(x, h - b + 12, text=f"{tmax * i / 5:.0f}", fill="#555")
            xy = []
            for tt, v in self.points:
                xy += [l + (w - l - r) * tt / tmax, h - b - (h - t - b) * v / vmax]
            if len(xy) >= 4:
                c.create_line(*xy, fill="#1a73e8", width=2)
            else:
                c.create_oval(xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3, fill="#1a73e8")
            avg = statistics.mean(p[1] for p in self.points)
            ya = h - b - (h - t - b) * avg / vmax
            c.create_line(l, ya, w - r, ya, fill="#e8710a", dash=(4, 3))
            c.create_text(w - r, ya - 8, text=f"середнє {avg:.2f}", anchor="e", fill="#e8710a")

    App().mainloop()


def main():
    mp.freeze_support()
    p = argparse.ArgumentParser(description=f"{APP_NAME} — визначення продуктивності ПК")
    p.add_argument("--cli", action="store_true", help="консольний режим")
    p.add_argument("--stress", type=int, default=0, help="тривалість стрес-тесту, с (0 — не виконувати)")
    p.add_argument("--threads", type=int, default=0, help="кількість потоків (за замовчуванням — усі ядра)")
    p.add_argument("--disk-mb", type=int, default=512, help="розмір тестового файлу, МБ")
    p.add_argument("--disk-dir", default=None, help="тека для тесту накопичувача")
    p.add_argument("--no-cpu", action="store_true")
    p.add_argument("--no-mem", action="store_true")
    p.add_argument("--no-disk", action="store_true")
    p.add_argument("--out", help="зберегти JSON-звіт у файл")
    args = p.parse_args()
    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
