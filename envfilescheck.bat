@echo off && chcp 65001

echo working dir is %cd%
echo downloading requirement aria2 check.
echo=
dir /a:d/b | findstr "aria2" > flag.txt
findstr "aria2" flag.txt >nul
if %errorlevel% ==0 (
    echo aria2 checked.
    echo=
) else (
    echo failed. please downloading aria2 from webpage!
    echo unzip it and put in this directory!
    timeout /T 5
    start https://github.com/aria2/aria2/releases/tag/release-1.36.0
    echo=
    goto end
)

echo envfiles checking start.
echo=

for /f %%x in ('findstr /i /c:"aria2" "flag.txt"') do (set aria2=%%x)&goto endSch
:endSch

set d32=f0D32k.pth
set d40=f0D40k.pth
set d48=f0D48k.pth
set g32=f0G32k.pth
set g40=f0G40k.pth
set g48=f0G48k.pth

set dld32=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth
set dld40=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth
set dld48=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth
set dlg32=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth
set dlg40=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth
set dlg48=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth

set hp2=HP2-人声vocals+非人声instrumentals.pth
set hp5=HP5-主旋律人声vocals+其他instrumentals.pth

set dlhp2=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth
set dlhp5=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth

set hb=hubert_base.pt

set dlhb=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt

echo dir check start.
echo=

if exist "%~dp0pretrained" (
        echo dir .\pretrained checked.
    ) else (
        echo failed. generating dir .\pretrained.
        mkdir pretrained
    )
if exist "%~dp0uvr5_weights" (
        echo dir .\uvr5_weights checked.
    ) else (
        echo failed. generating dir .\uvr5_weights.
        mkdir uvr5_weights
    )

echo=
echo dir check finished.

echo=
echo required files check start.

echo checking D32k.pth
if exist "%~dp0pretrained\D32k.pth" (
        echo D32k.pth in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth -d %~dp0pretrained -o D32k.pth
        if exist "%~dp0pretrained\D32k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking D40k.pth
if exist "%~dp0pretrained\D40k.pth" (
        echo D40k.pth in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth -d %~dp0pretrained -o D40k.pth
        if exist "%~dp0pretrained\D40k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
    echo checking D48k.pth
if exist "%~dp0pretrained\D48k.pth" (
        echo D48k.pth in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth -d %~dp0pretrained -o D48k.pth
        if exist "%~dp0pretrained\D48k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
    echo checking G32k.pth
if exist "%~dp0pretrained\G32k.pth" (
        echo G32k.pth in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth -d %~dp0pretrained -o G32k.pth
        if exist "%~dp0pretrained\G32k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
    echo checking G40k.pth
if exist "%~dp0pretrained\G40k.pth" (
        echo G40k.pth in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth -d %~dp0pretrained -o G40k.pth
        if exist "%~dp0pretrained\G40k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
    echo checking G48k.pth
if exist "%~dp0pretrained\G48k.pth" (
        echo G48k.pth in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth -d %~dp0pretrained -o G48k.pth
        if exist "%~dp0pretrained\G48k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )

echo checking %d32%
if exist "%~dp0pretrained\%d32%" (
        echo %d32% in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dld32% -d %~dp0pretrained -o %d32%
        if exist "%~dp0pretrained\%d32%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %d40%
if exist "%~dp0pretrained\%d40%" (
        echo %d40% in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dld40% -d %~dp0pretrained -o %d40%
        if exist "%~dp0pretrained\%d40%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %d48%
if exist "%~dp0pretrained\%d48%" (
        echo %d48% in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dld48% -d %~dp0pretrained -o %d48%
        if exist "%~dp0pretrained\%d48%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %g32%
if exist "%~dp0pretrained\%g32%" (
        echo %g32% in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlg32% -d %~dp0pretrained -o %g32%
        if exist "%~dp0pretrained\%g32%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %g40%
if exist "%~dp0pretrained\%g40%" (
        echo %g40% in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlg40% -d %~dp0pretrained -o %g40%
        if exist "%~dp0pretrained\%g40%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %g48%
if exist "%~dp0pretrained\%g48%" (
        echo %g48% in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlg48% -d %~dp0\pretrained -o %g48%
        if exist "%~dp0pretrained\%g48%" (echo download successful.) else (echo please try again!
        echo=)
    )

echo checking %hp2%
if exist "%~dp0uvr5_weights\%hp2%" (
        echo %hp2% in .\uvr5_weights checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlhp2% -d %~dp0\uvr5_weights -o %hp2%
        if exist "%~dp0uvr5_weights\%hp2%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %hp5%
if exist "%~dp0uvr5_weights\%hp5%" (
        echo %hp5% in .\uvr5_weights checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlhp5% -d %~dp0\uvr5_weights -o %HP5%
        if exist "%~dp0uvr5_weights\%hp5%" (echo download successful.) else (echo please try again!
        echo=)
    )

echo checking %hb%
if exist "%~dp0%hb%" (
        echo %hb% in .\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlhb% -d %~dp0 -o %hb%
        if exist "%~dp0%hb%" (echo download successful.) else (echo please try again!
        echo=)
    )

echo required files check finished.
echo envfiles check complete.
pause
:end
del flag.txt