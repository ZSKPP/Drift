@echo off
REM Generowanie strumieni AGRAWAL (abrupt, gradual, incremental) dla FRDD/FBDD
cd /d "%~dp0"

REM Pętla po seedach: 1,1,20 means from 1 to 20 with step 1
for /L %%S in (1,1,20) do (
    call :GENERUJ_DLA_SEEDA %%S
)

echo.
echo Gotowe. Powinny pojawic sie pliki:
echo   Agrawal_abrupt_10000_6500_seedX.arff
echo   Agrawal_gradual_10000_6500_seedX.arff
echo   Agrawal_incremental_10000_6500_seedX.arff
echo dla odpowiednich seedow.
echo.
pause
exit /b


:GENERUJ_DLA_SEEDA
set SEED=%1

echo.
echo AGRAWAL FEATURE DRIFT ABRUPT, seed %SEED%
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -i %SEED% -p 0.00 -f 1) -d (generators.AgrawalGenerator -i %SEED% -p 0.30 -f 1) -p 6500 -w 0) -f Agrawal_abrupt_10000_seed%SEED%.arff -m 10000"

echo.
echo AGRAWAL FEATURE DRIFT GRADUAL (w=2000), seed %SEED%
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -i %SEED% -p 0.00 -f 1) -d (generators.AgrawalGenerator -i %SEED% -p 0.30 -f 1) -p 6500 -w 2000) -f Agrawal_gradual_10000_seed%SEED%.arff -m 10000"

echo.
echo AGRAWAL FEATURE DRIFT INCREMENTAL (step=1000, w=1000), seed %SEED%
java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -i %SEED% -p 0.00 -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -i %SEED% -p 0.10 -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -i %SEED% -p 0.20 -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -i %SEED% -p 0.30 -f 1) -d (generators.AgrawalGenerator -i %SEED% -p 0.40 -f 1) -p 1000 -w 1000) -p 1000 -w 1000) -p 1000 -w 1000) -p 4500 -w 1000) -f Agrawal_incremental_10000_seed%SEED%.arff -m 10000"
%echo.
%exit /b


