; Установник PC Benchmark (Inno Setup 6)
; Збирається скриптом build.ps1 після PyInstaller; версію передає build.ps1 (/DAppVersion=...).

#define AppName "PC Benchmark"
#ifndef AppVersion
  #define AppVersion "0.10"
#endif
#define AppExe "PC Benchmark.exe"
#define AppUrl "https://github.com/zimbo812/pc-benchmark"

[Setup]
AppId={{8F3C2A51-6B7E-4C0D-9A11-3E5F2B7C9D40}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=zimbo812
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; установка без прав адміністратора (у %LOCALAPPDATA%\Programs); за бажанням — для всіх користувачів
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\build\installer
OutputBaseFilename=PC-Benchmark-Setup
SetupIconFile=..\assets\pcbench.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Languages]
Name: "ukrainian"; MessagesFile: "compiler:Languages\Ukrainian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
ukrainian.AutoStartGroup=Автозапуск:
english.AutoStartGroup=Startup:
ukrainian.AutoStart=Запускати PC Benchmark під час запуску Windows (у згорнутому вікні)
english.AutoStart=Start PC Benchmark when Windows starts (minimized)

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "{cm:AutoStart}"; GroupDescription: "{cm:AutoStartGroup}"; Flags: unchecked

[Files]
Source: "..\build\dist\PC Benchmark\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; автозапуск — запис у HKCU\...\Run; видаляється під час деінсталяції
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExe}"" --minimized"; Tasks: autostart; Flags: uninsdeletevalue
; якщо під час повторної установки галочку знято — прибрати старий запис
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "{#AppName}"; Tasks: not autostart; Flags: deletevalue

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
