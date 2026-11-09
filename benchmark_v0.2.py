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
VERSION = "0.2"

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


def compute_total(scores):
    """Загальний бал — зважене середнє геометричне балів категорій, що були виміряні."""
    keys = [k for k in WEIGHTS if scores.get(k)]
    wsum = sum(WEIGHTS[k] for k in keys)
    if not wsum:
        return 0.0
    return math.exp(sum(WEIGHTS[k] * math.log(max(scores[k], 1)) for k in keys) / wsum)


# ---------------------------------------------------------------------------
#  База порівняння: орієнтовні результати інших конфігурацій.
#  Значення перераховано у шкалу програми з відносної продуктивності
#  процесорів (однопотокові / багатопотокові тести), типу пам'яті та накопичувача
#  відносно виміряного еталонного ноутбука (Core i5-12450H). Це оцінки, а не заміри.
# ---------------------------------------------------------------------------

REFERENCE_PCS = [
    # назва,                       тип,        конфігурація,                          1 ядро, усі ядра, пам'ять, диск
    ("AMD Ryzen 9 9950X",          "Десктоп",  "16 ядер · DDR5-6000 · NVMe PCIe 5.0",   3318, 12572, 2360, 3213),
    ("Intel Core i9-14900K",       "Десктоп",  "24 ядра · DDR5-6400 · NVMe PCIe 4.0",   3413, 11943, 2433, 2785),
    ("AMD Ryzen 9 7950X",          "Десктоп",  "16 ядер · DDR5-6000 · NVMe PCIe 4.0",   2963, 11315, 2286, 2785),
    ("AMD Ryzen 7 7800X3D",        "Десктоп",  "8 ядер · DDR5-6000 · NVMe PCIe 4.0",    2678,  5657, 2212, 2570),
    ("Intel Core i7-13700H",       "Ноутбук",  "14 ядер · DDR5-4800 · NVMe PCIe 4.0",   2844,  5657, 1770, 2356),
    ("Intel Core Ultra 7 155H",    "Ноутбук",  "16 ядер · LPDDR5x · NVMe PCIe 4.0",     2607,  5343, 1844, 2356),
    ("Apple M2",                   "Ноутбук",  "8 ядер · LPDDR5 · NVMe",                2489,  2829, 2213, 2356),
    ("Intel Core i5-12400F",       "Десктоп",  "6 ядер · DDR4-3200 · NVMe PCIe 3.0",    2512,  3771, 1327, 1714),
    ("AMD Ryzen 5 5600X",          "Десктоп",  "6 ядер · DDR4-3600 · NVMe PCIe 3.0",    2370,  3457, 1254, 1714),
    ("Intel Core i7-8700K",        "Десктоп",  "6 ядер · DDR4-2666 · SATA SSD",         1778,  2829, 1033,  857),
    ("AMD Ryzen 5 5500U",          "Ноутбук",  "6 ядер · DDR4-3200 · NVMe PCIe 3.0",    1778,  2357, 1180, 1500),
    ("Intel Core i3-10100",        "Десктоп",  "4 ядра · DDR4-2666 · SATA SSD",         1706,  1572,  959,  857),
    ("Intel Core i5-8250U",        "Ноутбук",  "4 ядра · DDR4-2400 (1 канал) · SATA SSD", 1422, 1100,  664,  750),
    ("Intel Core i5-3470",         "Десктоп",  "4 ядра · DDR3-1600 · HDD 7200 об/хв",   1185,   943,  590,  171),
    ("Intel Celeron N4020",        "Ноутбук",  "2 ядра · DDR4-2400 · eMMC",              521,   189,  369,  214),
    ("Intel Core 2 Duo E8400",     "Десктоп",  "2 ядра · DDR2-800 · HDD 7200 об/хв",     427,   157,  221,  150),
]
REFERENCE_PCS = [
    {"name": n, "sub": f"{t} · {cfg}", "kind": "ref",
     "scores": {"cpu_single": a, "cpu_multi": b, "memory": c, "disk": d,
                "total": round(compute_total({"cpu_single": a, "cpu_multi": b, "memory": c, "disk": d}))}}
    for n, t, cfg, a, b, c, d in REFERENCE_PCS
]


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

        total = compute_total({k: v["score"] for k, v in self.categories.items()})
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


HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_history.json")


def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_history(items):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-50:], f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def pretty_cpu(name):
    for junk in ("(R)", "(TM)", "(tm)", " CPU", "  "):
        name = name.replace(junk, "" if junk != "  " else " ")
    name = name.replace("12th Gen ", "").replace("13th Gen ", "").replace("14th Gen ", "")
    return name.split(" @ ")[0].strip()


def report_to_entry(rep, own=True):
    """Перетворює JSON-звіт у запис для таблиці порівняння."""
    scores = {k: v["score"] for k, v in rep.get("categories", {}).items() if k in WEIGHTS}
    if len(scores) == len(WEIGHTS):
        scores["total"] = rep.get("total_score") or round(compute_total(scores))
    sysinfo = rep.get("system", {})
    sub = f"{'Ваш запуск' if own else 'Імпортовано'} · {rep.get('date', '')[:16]}"
    if sysinfo.get("ОЗП, ГБ"):
        sub += f" · {sysinfo['ОЗП, ГБ']} ГБ ОЗП"
    return {"id": f"{rep.get('date', '')}|{sysinfo.get('Процесор', '')}",
            "name": pretty_cpu(sysinfo.get("Процесор", "Невідомий ПК")),
            "sub": sub, "kind": "own" if own else "imported", "scores": scores}


def fmt(v):
    return f"{int(round(v)):,}".replace(",", " ")


def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # чіткі шрифти на HiDPI
        except Exception:
            pass

    C = {"bg": "#0f1117", "panel": "#161a24", "card": "#1d2230", "card2": "#232a3b", "border": "#2a3042",
         "text": "#e7e9f0", "text2": "#c3c8d6", "muted": "#8a90a2", "accent": "#4f8cff", "accent2": "#78a6ff",
         "green": "#3ecf8e", "orange": "#f5a524", "red": "#f25f5c", "track": "#262c3d"}
    FAM = "Segoe UI" if sys.platform == "win32" else "Helvetica"
    MONO = "Consolas" if sys.platform == "win32" else "Courier"
    CATS = [("cpu_single", "CPU · 1 ядро", "#4f8cff"), ("cpu_multi", "CPU · усі ядра", "#a78bfa"),
            ("memory", "Пам'ять", "#3ecf8e"), ("disk", "Накопичувач", "#f5a524")]
    METRICS = [("total", "Загальний бал")] + [(k, n) for k, n, _ in CATS]
    KIND = {"cur": ("Цей ПК", C["accent"]), "own": ("Ваші запуски", "#2fae7d"),
            "imported": ("Імпортовані звіти", "#a78bfa"), "ref": ("Еталонні системи", "#3d4866")}

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.S = self.winfo_fpixels("1i") / 96.0
            self.title(f"{APP_NAME} {VERSION} — визначення продуктивності ПК")
            self.configure(bg=C["bg"])
            self.geometry(f"{self.px(1220)}x{self.px(820)}")
            self.minsize(self.px(1060), self.px(720))
            self.q = queue.Queue()
            self.stop_event = threading.Event()
            self.report = None
            self.points = []
            self.live = {}
            self.history = load_history()
            own = [h for h in self.history if h.get("kind") == "own"]
            self.current = dict(own[-1], sub="Цей ПК · " + own[-1]["sub"]) if own else None
            self._styles()
            self._build()
            self.after(50, self._refresh_all)
            self.after(100, self._poll)

        def px(self, n):
            return int(round(n * self.S))

        # ================= стилі =================
        def _styles(self):
            px = self.px
            st = ttk.Style(self)
            st.theme_use("clam")
            st.configure(".", background=C["bg"], foreground=C["text"], fieldbackground=C["card"],
                         bordercolor=C["border"], lightcolor=C["border"], darkcolor=C["border"],
                         troughcolor=C["track"], font=(FAM, 10), insertcolor=C["text"],
                         selectbackground=C["accent"], selectforeground="white", arrowcolor=C["muted"])
            st.configure("Side.TCheckbutton", background=C["panel"], foreground=C["text"], padding=(0, px(3)),
                         indicatorbackground=C["card"], indicatorforeground="white",
                         upperbordercolor=C["border"], lowerbordercolor=C["border"])
            st.map("Side.TCheckbutton", background=[("active", C["panel"])],
                   indicatorbackground=[("selected", C["accent"]), ("active", C["card2"])])
            st.configure("Accent.TButton", background=C["accent"], foreground="white", borderwidth=0,
                         focusthickness=0, padding=(px(14), px(10)), font=(FAM, 10, "bold"),
                         lightcolor=C["accent"], darkcolor=C["accent"])
            st.map("Accent.TButton", background=[("disabled", "#2b3550"), ("pressed", "#3b74e0"), ("active", C["accent2"])],
                   foreground=[("disabled", C["muted"])], lightcolor=[("active", C["accent2"])], darkcolor=[("active", C["accent2"])])
            st.configure("Ghost.TButton", background=C["card"], foreground=C["text"], borderwidth=1,
                         bordercolor=C["border"], lightcolor=C["card"], darkcolor=C["card"],
                         focusthickness=0, padding=(px(12), px(7)))
            st.map("Ghost.TButton", background=[("disabled", C["panel"]), ("active", C["card2"])],
                   foreground=[("disabled", "#4f5566")])
            for w in ("TSpinbox", "TCombobox", "TEntry"):
                st.configure(w, fieldbackground=C["card"], foreground=C["text"], background=C["card"],
                             bordercolor=C["border"], lightcolor=C["card"], darkcolor=C["card"], padding=px(4))
            st.map("TCombobox", fieldbackground=[("readonly", C["card"])], foreground=[("readonly", C["text"])],
                   selectbackground=[("readonly", C["card"])], selectforeground=[("readonly", C["text"])])
            self.option_add("*TCombobox*Listbox.background", C["card"])
            self.option_add("*TCombobox*Listbox.foreground", C["text"])
            self.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
            st.configure("Accent.Horizontal.TProgressbar", background=C["accent"], troughcolor=C["track"],
                         bordercolor=C["track"], lightcolor=C["accent"], darkcolor=C["accent"], thickness=px(6))
            st.configure("TNotebook", background=C["bg"], borderwidth=0, tabmargins=0, bordercolor=C["panel"],
                         lightcolor=C["panel"], darkcolor=C["panel"])
            st.configure("TNotebook.Tab", background=C["bg"], foreground=C["muted"], borderwidth=0,
                         padding=(px(16), px(8)), font=(FAM, 10, "bold"), lightcolor=C["bg"],
                         bordercolor=C["bg"], darkcolor=C["bg"], focuscolor=C["panel"])
            st.map("TNotebook.Tab", background=[("selected", C["panel"]), ("active", C["card"])],
                   foreground=[("selected", C["text"]), ("active", C["text2"])],
                   lightcolor=[("selected", C["accent"])], bordercolor=[("selected", C["panel"])],
                   darkcolor=[("selected", C["panel"])], padding=[("selected", (px(16), px(8)))],
                   expand=[("selected", (0, 0, 0, 0))])
            st.configure("Treeview", background=C["panel"], fieldbackground=C["panel"], foreground=C["text2"],
                         rowheight=px(30), borderwidth=0, font=(FAM, 10))
            st.map("Treeview", background=[("selected", "#24324f")], foreground=[("selected", C["text"])])
            st.configure("Treeview.Heading", background=C["card"], foreground=C["muted"], relief="flat",
                         font=(FAM, 9, "bold"), padding=(px(8), px(6)), borderwidth=0)
            st.map("Treeview.Heading", background=[("active", C["card2"])])
            st.configure("Vertical.TScrollbar", background=C["card"], troughcolor=C["panel"],
                         bordercolor=C["panel"], lightcolor=C["card"], darkcolor=C["card"], arrowsize=px(12),
                         gripcount=0)
            st.map("Vertical.TScrollbar", background=[("disabled", C["panel"]), ("active", C["card2"]),
                                                      ("!active", C["card"])],
                   lightcolor=[("disabled", C["panel"])], darkcolor=[("disabled", C["panel"])],
                   arrowcolor=[("disabled", C["border"])])
            st.layout("Seg.TRadiobutton", st.layout("Toolbutton"))
            st.configure("Seg.TRadiobutton", background=C["card"], foreground=C["text2"], borderwidth=0,
                         padding=(px(12), px(6)), font=(FAM, 9, "bold"), anchor="center")
            st.map("Seg.TRadiobutton", background=[("selected", C["accent"]), ("active", C["card2"])],
                   foreground=[("selected", "white")])

        # ================= побудова вікна =================
        def _build(self):
            px = self.px
            hdr = tk.Frame(self, bg=C["panel"])
            hdr.pack(fill="x")
            logo = tk.Canvas(hdr, width=px(38), height=px(38), bg=C["panel"], highlightthickness=0)
            logo.pack(side="left", padx=(px(18), px(10)), pady=px(12))
            logo.create_oval(1, 1, px(37), px(37), fill=C["accent"], outline="")
            logo.create_text(px(19), px(19), text="⚡", fill="white", font=(FAM, 14, "bold"))
            title = tk.Frame(hdr, bg=C["panel"])
            title.pack(side="left")
            tk.Label(title, text=APP_NAME, bg=C["panel"], fg=C["text"], font=(FAM, 15, "bold")).pack(anchor="w")
            tk.Label(title, text="Визначення продуктивності ПК з використанням алгоритмів стрес-тестування",
                     bg=C["panel"], fg=C["muted"], font=(FAM, 9)).pack(anchor="w")
            info = system_info()
            right = tk.Frame(hdr, bg=C["panel"])
            right.pack(side="right", padx=px(18))
            tk.Label(right, text=pretty_cpu(str(info["Процесор"])), bg=C["panel"], fg=C["text"],
                     font=(FAM, 10, "bold")).pack(anchor="e")
            tk.Label(right, text=f"{info['Логічних ядер']} потоків  ·  {info['ОЗП, ГБ']} ГБ ОЗП  ·  "
                                 f"{info['ОС']}  ·  Python {info['Python']}",
                     bg=C["panel"], fg=C["muted"], font=(FAM, 9)).pack(anchor="e")
            tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

            body = tk.Frame(self, bg=C["bg"])
            body.pack(fill="both", expand=True)
            side = tk.Frame(body, bg=C["panel"], width=px(300))
            side.pack(side="left", fill="y")
            side.pack_propagate(False)
            tk.Frame(body, bg=C["border"], width=1).pack(side="left", fill="y")
            main = tk.Frame(body, bg=C["bg"])
            main.pack(side="left", fill="both", expand=True, padx=px(18), pady=px(16))
            self._build_side(side)
            self._build_main(main)

        def _section(self, parent, text):
            tk.Label(parent, text=text, bg=C["panel"], fg=C["muted"], font=(FAM, 8, "bold")).pack(
                anchor="w", padx=self.px(20), pady=(self.px(14), self.px(4)))

        def _build_side(self, p):
            px = self.px
            pad = px(20)
            btns = tk.Frame(p, bg=C["panel"])
            btns.pack(side="bottom", fill="x", padx=pad, pady=pad)
            self.btn_start = ttk.Button(btns, text="▶   Запустити тест", style="Accent.TButton", command=self.start)
            self.btn_start.pack(fill="x")
            row = tk.Frame(btns, bg=C["panel"])
            row.pack(fill="x", pady=(px(8), 0))
            row.columnconfigure((0, 1), weight=1, uniform="b")
            self.btn_stop = ttk.Button(row, text="■  Стоп", style="Ghost.TButton", command=self.stop, state="disabled")
            self.btn_stop.grid(row=0, column=0, sticky="we", padx=(0, px(4)))
            self.btn_save = ttk.Button(row, text="Зберегти звіт", style="Ghost.TButton", command=self.save,
                                       state="disabled")
            self.btn_save.grid(row=0, column=1, sticky="we", padx=(px(4), 0))

            self.gauge = tk.Canvas(p, width=px(230), height=px(200), bg=C["panel"], highlightthickness=0)
            self.gauge.pack(pady=(px(14), 0))
            self.lbl_status = tk.Label(p, text="Готово до запуску", bg=C["panel"], fg=C["muted"],
                                       font=(FAM, 9), wraplength=px(255))
            self.lbl_status.pack(pady=(0, px(6)))
            self.progress = ttk.Progressbar(p, style="Accent.Horizontal.TProgressbar", maximum=100)
            self.progress.pack(fill="x", padx=pad)

            self._section(p, "ТЕСТИ")
            self.v = {k: tk.BooleanVar(value=True) for k in ("cpu_single", "cpu_multi", "memory", "disk", "stress")}
            for k, t in (("cpu_single", "Процесор — одне ядро"), ("cpu_multi", "Процесор — усі ядра"),
                         ("memory", "Оперативна пам'ять"), ("disk", "Накопичувач"),
                         ("stress", "Стрес-тест стабільності")):
                ttk.Checkbutton(p, text=t, variable=self.v[k], style="Side.TCheckbutton").pack(anchor="w", padx=pad)

            self._section(p, "ПАРАМЕТРИ")
            g = tk.Frame(p, bg=C["panel"])
            g.pack(fill="x", padx=pad)
            g.columnconfigure(0, weight=1)
            self.v_threads = tk.IntVar(value=os.cpu_count())
            self.v_stress = tk.IntVar(value=60)
            self.v_disk = tk.IntVar(value=512)
            self.v_dir = tk.StringVar(value=tempfile.gettempdir())
            widgets = [
                ("Потоків", ttk.Spinbox(g, from_=1, to=(os.cpu_count() or 1) * 2, width=7, textvariable=self.v_threads)),
                ("Стрес-тест, с", ttk.Spinbox(g, from_=10, to=3600, increment=10, width=7, textvariable=self.v_stress)),
                ("Файл диска, МБ", ttk.Combobox(g, values=[128, 256, 512, 1024, 2048], width=6, state="readonly",
                                                textvariable=self.v_disk)),
            ]
            for i, (label, w) in enumerate(widgets):
                tk.Label(g, text=label, bg=C["panel"], fg=C["text2"], font=(FAM, 10)).grid(row=i, column=0, sticky="w", pady=px(3))
                w.grid(row=i, column=1, sticky="e", pady=px(3))
            tk.Label(p, text="Тека для тесту диска", bg=C["panel"], fg=C["text2"], font=(FAM, 10)).pack(
                anchor="w", padx=pad, pady=(px(6), px(2)))
            d = tk.Frame(p, bg=C["panel"])
            d.pack(fill="x", padx=pad)
            ttk.Entry(d, textvariable=self.v_dir).pack(side="left", fill="x", expand=True)
            ttk.Button(d, text="…", width=3, style="Ghost.TButton", command=self._choose_dir).pack(side="left", padx=(px(6), 0))

        def _build_main(self, p):
            px = self.px
            cards = tk.Frame(p, bg=C["bg"])
            cards.pack(fill="x")
            self.cards = {}
            for i, (k, name, color) in enumerate(CATS):
                cards.columnconfigure(i, weight=1, uniform="c")
                f = tk.Frame(cards, bg=C["card"], highlightthickness=1, highlightbackground=C["border"])
                f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else px(6), 0))
                tk.Frame(f, bg=color, height=px(3)).pack(fill="x")
                tk.Label(f, text=name.upper(), bg=C["card"], fg=C["muted"], font=(FAM, 8, "bold")).pack(
                    anchor="w", padx=px(14), pady=(px(10), 0))
                val = tk.Label(f, text="—", bg=C["card"], fg=C["text"], font=(FAM, 22, "bold"))
                val.pack(anchor="w", padx=px(14))
                sub = tk.Label(f, text="ще не виміряно", bg=C["card"], fg=C["muted"], font=(FAM, 9))
                sub.pack(anchor="w", padx=px(14))
                bar = tk.Canvas(f, height=px(6), bg=C["card"], highlightthickness=0)
                bar.pack(fill="x", padx=px(14), pady=(px(8), px(14)))
                bar.bind("<Configure>", lambda e: self._refresh_cards())
                self.cards[k] = (val, sub, bar, color)

            self.nb = ttk.Notebook(p)
            self.nb.pack(fill="both", expand=True, pady=(px(16), 0))
            self._tab_results()
            self._tab_stress()
            self._tab_compare()
            self._tab_log()

        def _tab_results(self):
            t = tk.Frame(self.nb, bg=C["panel"])
            self.nb.add(t, text="Результати")
            cols = ("category", "test", "value", "unit", "score")
            self.tree = ttk.Treeview(t, columns=cols, show="headings")
            for c, title, w, a in (("category", "КАТЕГОРІЯ", 140, "w"), ("test", "ТЕСТ", 300, "w"),
                                   ("value", "ЗНАЧЕННЯ", 120, "e"), ("unit", "ОД.", 70, "w"),
                                   ("score", "БАЛИ", 90, "e")):
                self.tree.heading(c, text=title, anchor=a)
                self.tree.column(c, width=self.px(w), anchor=a)
            self.tree.tag_configure("cat", background=C["card2"], foreground=C["accent2"], font=(FAM, 10, "bold"))
            self.tree.tag_configure("odd", background="#191d29")
            sb = ttk.Scrollbar(t, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")

        def _tab_stress(self):
            px = self.px
            t = tk.Frame(self.nb, bg=C["panel"])
            self.nb.add(t, text="Стрес-тест")
            stats = tk.Frame(t, bg=C["panel"])
            stats.pack(fill="x", padx=px(14), pady=(px(12), 0))
            self.st_lbl = {}
            for i, (k, title) in enumerate((("errors", "ПОМИЛКИ ОБЧИСЛЕНЬ"), ("deg", "ПАДІННЯ ПРОДУКТИВНОСТІ"),
                                            ("cv", "КОЕФ. ВАРІАЦІЇ"), ("verdict", "ВИСНОВОК"))):
                stats.columnconfigure(i, weight=3 if k == "verdict" else 2, uniform="s")
                f = tk.Frame(stats, bg=C["card"])
                f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else px(6), 0))
                tk.Label(f, text=title, bg=C["card"], fg=C["muted"], font=(FAM, 8, "bold")).pack(anchor="w", padx=px(12), pady=(px(8), 0))
                lbl = tk.Label(f, text="—", bg=C["card"], fg=C["text"], font=(FAM, 12, "bold"), wraplength=px(260), justify="left")
                lbl.pack(anchor="w", padx=px(12), pady=(0, px(8)))
                self.st_lbl[k] = lbl
            self.canvas = tk.Canvas(t, bg=C["panel"], highlightthickness=0)
            self.canvas.pack(fill="both", expand=True, padx=px(8), pady=px(8))
            self.canvas.bind("<Configure>", lambda e: self._draw_chart())

        def _tab_compare(self):
            px = self.px
            t = tk.Frame(self.nb, bg=C["panel"])
            self.nb.add(t, text="Порівняння")
            top = tk.Frame(t, bg=C["panel"])
            top.pack(fill="x", padx=px(14), pady=(px(12), px(6)))
            self.v_metric = tk.StringVar(value="total")
            for key, name in METRICS:
                ttk.Radiobutton(top, text=name, value=key, variable=self.v_metric, style="Seg.TRadiobutton",
                                command=self._draw_compare).pack(side="left", padx=(0, px(4)))
            ttk.Button(top, text="Очистити історію", style="Ghost.TButton", command=self.clear_history).pack(side="right")
            ttk.Button(top, text="＋ Додати звіт іншого ПК…", style="Ghost.TButton",
                       command=self.import_report).pack(side="right", padx=px(6))

            info = tk.Frame(t, bg=C["panel"])
            info.pack(fill="x", padx=px(14), pady=(px(2), px(6)))
            self.lbl_cmp = tk.Label(info, text="", bg=C["panel"], fg=C["text"], font=(FAM, 10, "bold"), anchor="w")
            self.lbl_cmp.pack(side="left")
            legend = tk.Frame(t, bg=C["panel"])
            legend.pack(fill="x", padx=px(14), pady=(0, px(6)))
            for kind in ("cur", "own", "imported", "ref"):
                name, color = KIND[kind]
                sq = tk.Canvas(legend, width=px(10), height=px(10), bg=C["panel"], highlightthickness=0)
                sq.create_rectangle(0, 0, px(10), px(10), fill=color, outline="")
                sq.pack(side="left")
                tk.Label(legend, text=name, bg=C["panel"], fg=C["muted"], font=(FAM, 9)).pack(side="left", padx=(px(5), px(16)))

            note = tk.Label(t, text="Еталонні системи — орієнтовні оцінки, перераховані у шкалу програми з відносної "
                                    "продуктивності процесорів, пам'яті та накопичувачів. Точні цифри дає лише "
                                    "запуск програми на відповідному ПК — додайте його JSON-звіт кнопкою вище.",
                            bg=C["panel"], fg=C["muted"], font=(FAM, 8), wraplength=px(820), justify="left")
            note.pack(side="bottom", fill="x", padx=px(14), pady=(px(4), px(10)))
            note.bind("<Configure>", lambda e: note.config(wraplength=e.width - 10))
            area = tk.Frame(t, bg=C["panel"])
            area.pack(fill="both", expand=True, padx=(px(8), 0))
            self.cmp = tk.Canvas(area, bg=C["panel"], highlightthickness=0, yscrollincrement=px(24))
            sb = ttk.Scrollbar(area, orient="vertical", command=self.cmp.yview)
            self.cmp.configure(yscrollcommand=sb.set)
            self.cmp.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            self.cmp.bind("<Configure>", lambda e: self._draw_compare())
            self.cmp.bind("<Enter>", lambda e: self.cmp.bind_all(
                "<MouseWheel>", lambda ev: self.cmp.yview_scroll(int(-ev.delta / 120), "units")))
            self.cmp.bind("<Leave>", lambda e: self.cmp.unbind_all("<MouseWheel>"))

        def _tab_log(self):
            t = tk.Frame(self.nb, bg=C["panel"])
            self.nb.add(t, text="Журнал")
            self.log = tk.Text(t, font=(MONO, 9), wrap="word", bg=C["panel"], fg=C["text2"], relief="flat",
                               insertbackground=C["text"], padx=self.px(12), pady=self.px(10),
                               selectbackground=C["accent"], borderwidth=0)
            self.log.pack(fill="both", expand=True)
            self.log.tag_configure("head", foreground=C["accent2"], font=(MONO, 9, "bold"))
            self.log.tag_configure("time", foreground="#5b6175")

        def _choose_dir(self):
            d = filedialog.askdirectory(initialdir=self.v_dir.get())
            if d:
                self.v_dir.set(d)

        # ================= дані для відображення =================
        def _scores(self):
            if self.live:
                return self.live
            return self.current["scores"] if self.current else {}

        def _total(self):
            s = self._scores()
            return compute_total(s), sum(1 for k in WEIGHTS if s.get(k)) < len(WEIGHTS)

        def _compare_rows(self, metric):
            rows = [dict(r) for r in REFERENCE_PCS]
            cur_id = self.current["id"] if self.current else None
            rows += [dict(h) for h in self.history if h.get("id") != cur_id]
            if self.current:
                rows.append(dict(self.current, kind="cur"))
            rows = [r for r in rows if r["scores"].get(metric)]
            rows.sort(key=lambda r: r["scores"][metric], reverse=True)
            return rows

        def _rank(self, metric, value):
            refs = [r["scores"][metric] for r in REFERENCE_PCS]
            better = sum(1 for v in refs if v < value)
            return 1 + sum(1 for v in refs if v > value), len(refs) + 1, round(100 * better / len(refs))

        # ================= відмальовування =================
        def _refresh_all(self):
            self._draw_gauge()
            self._refresh_cards()
            self._draw_compare()
            self._draw_chart()

        def _draw_gauge(self):
            px = self.px
            c = self.gauge
            c.delete("all")
            w = int(c["width"])
            m, th = px(22), px(14)
            size = w - 2 * m
            box = (m, m, m + size, m + size)
            c.create_arc(*box, start=225, extent=-270, style="arc", width=th, outline=C["track"])
            total, partial = self._total()
            cx, cy = w / 2, m + size / 2
            sub, color = "запустіть тест", C["muted"]
            if total:
                vmax = max(r["scores"]["total"] for r in REFERENCE_PCS)
                _, _, pct = self._rank("total", total)
                color = C["green"] if pct >= 70 else C["accent"] if pct >= 35 else C["orange"]
                frac = max(0.01, min(total / vmax, 1.0))
                # градієнт дуги: кілька сегментів від темнішого до основного кольору
                steps = 24
                for i in range(steps):
                    a0 = 225 - 270 * frac * i / steps
                    seg = -270 * frac / steps - 0.6
                    c.create_arc(*box, start=a0, extent=seg, style="arc", width=th,
                                 outline=self._mix(C["track"], color, 0.35 + 0.65 * (i + 1) / steps))
                sub = "частковий результат" if partial else f"краще за {pct}% систем"
            c.create_text(cx, cy - px(6), text=fmt(total) if total else "—", fill=C["text"], font=(FAM, 28, "bold"))
            c.create_text(cx, cy + px(26), text="ЗАГАЛЬНИЙ БАЛ", fill=C["muted"], font=(FAM, 8, "bold"))
            c.create_text(cx, cy + px(46), text=sub, fill=color, font=(FAM, 9, "bold"))

        @staticmethod
        def _mix(c1, c2, t):
            a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
            b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
            return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))

        def _refresh_cards(self):
            s = self._scores()
            for k, (val, sub, bar, color) in self.cards.items():
                bar.delete("all")
                w = bar.winfo_width()
                bar.create_rectangle(0, 0, w, self.px(6), fill=C["track"], outline="")
                v = s.get(k)
                if not v:
                    val.config(text="—")
                    sub.config(text="ще не виміряно")
                    continue
                vmax = max(r["scores"][k] for r in REFERENCE_PCS)
                rank, n, pct = self._rank(k, v)
                val.config(text=fmt(v))
                sub.config(text=f"#{rank} з {n} · краще за {pct}% систем")
                bar.create_rectangle(0, 0, max(self.px(4), w * min(v / vmax, 1)), self.px(6), fill=color, outline="")

        def _draw_compare(self):
            px = self.px
            c = self.cmp
            c.delete("all")
            metric = self.v_metric.get()
            rows = self._compare_rows(metric)
            W = max(c.winfo_width(), px(640))
            rh = px(50)
            x_name, x_bar, right = px(52), px(380), px(80)
            bar_w = W - x_bar - right
            vmax = max((r["scores"][metric] for r in rows), default=1)
            for i, r in enumerate(rows):
                y0, y1 = i * rh + px(4), (i + 1) * rh
                cy = (y0 + y1) / 2
                kind = r.get("kind", "ref")
                color = KIND[kind][1]
                if kind == "cur":
                    c.create_rectangle(px(4), y0, W - px(6), y1, fill="#18233d", outline=C["accent"])
                elif i % 2:
                    c.create_rectangle(px(4), y0, W - px(6), y1, fill="#191d29", outline="")
                c.create_text(px(28), cy, text=str(i + 1), fill=C["accent2"] if kind == "cur" else C["muted"],
                              font=(FAM, 10, "bold"))
                name = r["name"] if len(r["name"]) <= 40 else r["name"][:39] + "…"
                c.create_text(x_name, cy - px(8), text=name, anchor="w", font=(FAM, 10, "bold"),
                              fill=C["text"] if kind != "ref" else C["text2"])
                c.create_text(x_name, cy + px(10), text=r["sub"], anchor="w", fill=C["muted"], font=(FAM, 8))
                v = r["scores"][metric]
                c.create_rectangle(x_bar, cy - px(7), x_bar + bar_w, cy + px(7), fill=C["card"], outline="")
                c.create_rectangle(x_bar, cy - px(7), x_bar + max(px(3), bar_w * v / vmax), cy + px(7),
                                   fill=color, outline="")
                c.create_text(x_bar + bar_w + px(12), cy, text=fmt(v), anchor="w", font=(FAM, 10, "bold"),
                              fill=C["text"] if kind != "ref" else C["text2"])
            total_h = len(rows) * rh + px(10)
            c.configure(scrollregion=(0, 0, W, total_h))
            cur_idx = next((i for i, r in enumerate(rows) if r.get("kind") == "cur"), None)
            if cur_idx is not None and total_h > c.winfo_height():
                c.yview_moveto(max(0.0, (cur_idx * rh - c.winfo_height() / 2 + rh / 2) / total_h))

            cur = self.current["scores"].get(metric) if self.current else None
            name = dict(METRICS)[metric]
            if cur:
                rank, n, pct = self._rank(metric, cur)
                self.lbl_cmp.config(text=f"Ваш ПК: {fmt(cur)} ({name}) — {rank} місце з {n}, "
                                         f"швидший за {pct}% еталонних систем", fg=C["text"])
            elif self.current and metric == "total":
                self.lbl_cmp.config(text="Для загального бала запустіть усі 4 основні тести", fg=C["orange"])
            else:
                self.lbl_cmp.config(text="Запустіть тест, щоб побачити свій ПК у рейтингу", fg=C["muted"])

        def _draw_chart(self):
            px = self.px
            c = self.canvas
            c.delete("all")
            w, h = c.winfo_width(), c.winfo_height()
            if w < 100 or h < 100:
                return
            l, r, t, b = px(56), px(20), px(28), px(36)
            c.create_text(l, px(12), text="Пропускна здатність, завдань/с", anchor="w", fill=C["muted"], font=(FAM, 9))
            if not self.points:
                c.create_rectangle(l, t, w - r, h - b, outline=C["border"], dash=(3, 4))
                c.create_text(w / 2, h / 2, text="Графік з'явиться під час стрес-тесту", fill=C["muted"], font=(FAM, 11))
                return
            tmax = max(self.points[-1][0], 1)
            vmax = max(p[1] for p in self.points) * 1.15
            for i in range(5):
                y = h - b - (h - t - b) * i / 4
                c.create_line(l, y, w - r, y, fill=C["track"])
                c.create_text(l - px(8), y, text=f"{vmax * i / 4:.0f}", anchor="e", fill=C["muted"], font=(FAM, 8))
            for i in range(6):
                x = l + (w - l - r) * i / 5
                c.create_text(x, h - b + px(14), text=f"{tmax * i / 5:.0f} с", fill=C["muted"], font=(FAM, 8))
            xy = []
            for tt, v in self.points:
                xy += [l + (w - l - r) * tt / tmax, h - b - (h - t - b) * v / vmax]
            if len(xy) >= 4:
                c.create_polygon(xy[0], h - b, *xy, xy[-2], h - b, fill="#1a2742", outline="")
                c.create_line(*xy, fill=C["accent"], width=px(2), joinstyle="round")
            c.create_oval(xy[-2] - px(4), xy[-1] - px(4), xy[-2] + px(4), xy[-1] + px(4), fill=C["accent"], outline="white")
            avg = statistics.mean(p[1] for p in self.points)
            ya = h - b - (h - t - b) * avg / vmax
            c.create_line(l, ya, w - r, ya, fill=C["orange"], dash=(5, 4))
            c.create_text(l + px(8), ya - px(10), text=f"середнє {avg:.1f}", anchor="w", fill=C["orange"], font=(FAM, 9, "bold"))

        # ================= керування =================
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
            for lbl in self.st_lbl.values():
                lbl.config(text="—", fg=C["text"])
            self.points, self.live, self.report = [], {}, None
            self.progress["value"] = 0
            self._refresh_all()
            self.stop_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_save.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.nb.select(0)
            bench = Benchmark(cfg, lambda k, d: self.q.put((k, d)), self.stop_event)
            threading.Thread(target=bench.run, daemon=True).start()

        def stop(self):
            self.stop_event.set()
            self.lbl_status.config(text="Зупинка після поточного кроку…")

        def save(self):
            if not self.report:
                return
            path = filedialog.asksaveasfilename(
                defaultextension=".json", initialfile=f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}",
                filetypes=[("JSON — для порівняння на іншому ПК", "*.json"), ("Текстовий звіт", "*.txt")])
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                if path.lower().endswith(".txt"):
                    f.write(report_to_text(self.report))
                else:
                    json.dump(self.report, f, ensure_ascii=False, indent=2)
            self.lbl_status.config(text=f"Звіт збережено: {os.path.basename(path)}")

        def import_report(self):
            paths = filedialog.askopenfilenames(title="JSON-звіти PC Benchmark з інших комп'ютерів",
                                                filetypes=[("JSON", "*.json")])
            added = 0
            for path in paths:
                try:
                    with open(path, encoding="utf-8") as f:
                        rep = json.load(f)
                    entry = report_to_entry(rep, own=False)
                except (OSError, ValueError, AttributeError, KeyError, TypeError):
                    messagebox.showerror(APP_NAME, f"Не вдалося прочитати звіт:\n{path}")
                    continue
                if not entry["scores"]:
                    messagebox.showwarning(APP_NAME, f"У звіті немає результатів:\n{path}")
                    continue
                if any(h["id"] == entry["id"] for h in self.history) or (self.current and self.current["id"] == entry["id"]):
                    continue
                self.history.append(entry)
                added += 1
            if added:
                save_history(self.history)
                self._draw_compare()
                self.lbl_status.config(text=f"Додано звітів до порівняння: {added}")

        def clear_history(self):
            if not self.history:
                return
            if messagebox.askyesno(APP_NAME, "Видалити всі збережені запуски та імпортовані звіти з порівняння?"):
                self.history = []
                save_history(self.history)
                if not self.report:
                    self.current = None
                self._refresh_all()

        # ================= події з робочого потоку =================
        def _poll(self):
            try:
                while True:
                    kind, data = self.q.get_nowait()
                    getattr(self, f"_on_{kind}")(data)
            except queue.Empty:
                pass
            self.after(100, self._poll)

        def _on_log(self, msg):
            self.log.insert("end", time.strftime("%H:%M:%S  "), "time")
            self.log.insert("end", msg + "\n", "head" if msg.startswith("==") else ())
            self.log.see("end")
            if not msg.startswith("  "):
                self.lbl_status.config(text=msg.strip("= "))

        def _on_progress(self, p):
            self.progress["value"] = p

        def _on_result(self, r):
            n = len(self.tree.get_children())
            self.tree.insert("", "end", values=(r["category"], r["test"], f"{r['value']:,.2f}".replace(",", " "),
                                                r["unit"], fmt(r["score"]) if r["score"] else ""),
                             tags=("odd",) if n % 2 else ())

        def _on_category(self, c):
            self.tree.insert("", "end", values=("", f"Підсумок: {c['name']}", "", "", fmt(c["score"])), tags=("cat",))
            self.live[c["key"]] = c["score"]
            self._draw_gauge()
            self._refresh_cards()

        def _on_stress_point(self, p):
            self.points.append(p)
            if len(self.points) == 1:
                self.nb.select(1)
            self._draw_chart()

        def _on_stress_done(self, s):
            self.st_lbl["errors"].config(text=str(s["errors"]), fg=C["red"] if s["errors"] else C["green"])
            self.st_lbl["deg"].config(text=f"{s['degradation_pct']}%",
                                      fg=C["orange"] if s["degradation_pct"] > 10 else C["green"])
            self.st_lbl["cv"].config(text=f"{s['variation_pct']}%")
            ok = s["verdict"] == "Стабільно"
            self.st_lbl["verdict"].config(text=s["verdict"], fg=C["green"] if ok else
                                          C["red"] if s["errors"] else C["orange"])

        def _on_done(self, rep):
            self.report = rep
            self.lbl_status.config(text=f"{rep['status']} за {rep['duration_sec']} с")
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.btn_save.config(state="normal")
            entry = report_to_entry(rep, own=True)
            if entry["scores"]:
                if rep["status"] == "Завершено":
                    self.history.append(entry)
                    save_history(self.history)
                self.current = dict(entry, sub="Цей ПК · щойно виміряно")
            self._refresh_all()

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
