@echo off
echo ======================================================
echo   TrafficAI: Alienware RTX 5090 Environment Setup
echo ======================================================

:: Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден. Пожалуйста, установите Python 3.10+ и добавьте его в PATH.
    pause
    exit /b
)

echo [1/3] Создание виртуального окружения (venv)...
python -m venv venv

echo [2/3] Активация окружения и обновление pip...
call .\venv\Scripts\activate
python -m pip install --upgrade pip

echo [3/3] Установка всех библиотек из requirements.txt (включая CUDA-версию PyTorch)...
pip install -r requirements.txt

echo ======================================================
echo   УСТАНОВКА ЗАВЕРШЕНА! 
echo   Ваша RTX 5090 готова к Фазе 8.
echo   Для старта обучения запустите: .\marathon.ps1
echo ======================================================
pause