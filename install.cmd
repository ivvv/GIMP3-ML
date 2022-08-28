@powershell -ExecutionPolicy Bypass -Command "$_=((Get-Content \"%~f0\") -join \"`n\");iex $_.Substring($_.IndexOf(\"goto :\"+\"EOF\")+9)"
@goto :EOF

param([switch]$cpuonly = $false)

echo "-----------Installing GIMP-ML-----------"

python -m pip install virtualenv
python -m virtualenv gimpenv3
if (!((Get-Command python).Path | Select-String -Pattern gimpenv3 -Quiet)) {
    throw "Failed to activate the created environment."
}
if ($cpuonly) {
    gimpenv3\Scripts\python.exe -m pip install torch torchvision torchaudio
} else {
    gimpenv3\Scripts\python.exe -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    gimpenv3\Scripts\python.exe -m pip install cudatoolkit
}

gimpenv3\Scripts\python.exe -m pip install -r requirements.txt
gimpenv3\Scripts\python.exe -m pip install -e .
gimpenv3\Scripts\python.exe gimpml/init_config.py

# Register the plugins directory in GIMP settings
$pluginsDir = '$($PSScriptRoot)\plugins'
$gimpdir = (gci -Filter "GIMP*" -Directory -ErrorAction SilentlyContinue -Path "C:\Program Files\").FullName
$gimp = (dir  "$($gimpdir)\bin\gimp-console-*.exe").FullName
if (!($gimp -and (Test-Path $gimp))) {
    throw "Could not find GIMP! You will have to add '$pluginsDir' to Preferences -> Folders -> Plug-ins manually."
}
$version = (& $gimp --version | Select-String -Pattern [[32]\.\d+).Matches.Value
if (!($version)) {
    throw "Could not determine GIMP version."
}
$gimprcPath = ($env:APPDATA + '\GIMP\' + $version + '\gimprc')
$escapedDir = [regex]::escape($pluginsDir)
if (!(Test-Path $gimprcPath)) {
    New-Item $gimprcPath -Force
}
if (!(Select-String -Path $gimprcPath -Pattern 'plug-in-path' -Quiet)) {
    (cat $gimprcPath) + ('(plug-in-path "${gimp_dir}\\plug-ins;${gimp_plug_in_dir}\\plug-ins;' + $escapedDir + '")') | Set-Content $gimprcPath
} elseif (!(Select-String -Path $gimprcPath -Pattern ([regex]::escape($escapedDir)) -Quiet)) {
    (cat $gimprcPath) -replace '\(\s*plug-in-path\s+"', ('$0' + $escapedDir + ';') | Set-Content $gimprcPath
}

echo "-----------Installed GIMP-ML------------"
