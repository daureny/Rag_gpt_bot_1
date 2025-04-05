<?php
/**
 * Файл: chatbot-proxy.php
 * Описание: Прокси для запросов к чат-боту FastAPI
 */

// Настройки CORS для предотвращения проблем с доступом
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST, GET, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, admin-token");

// Если это предварительный запрос OPTIONS, завершаем выполнение
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit(0);
}

// Настройки
$bot_api_url = 'https://2f93-2a03-32c0-2d-d051-716a-650e-df98-8a9f.ngrok-free.app'; // URL вашего FastAPI сервера без '/ask' в конце
$admin_password = '19091979'; // Замените на пароль администратора (должен совпадать с ADMIN_PASSWORD в .env файле бота)

// Предотвращаем кэширование
header('Cache-Control: no-store, no-cache, must-revalidate, max-age=0');
header('Cache-Control: post-check=0, pre-check=0', false);
header('Pragma: no-cache');
header('Content-Type: application/json; charset=utf-8');

// Проверяем метод запроса
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode([
        'status' => 'error',
        'message' => 'Неверный метод запроса'
    ]);
    exit;
}

// Проверяем наличие параметра действия
$action = isset($_POST['action']) ? $_POST['action'] : 'ask';

switch ($action) {
    case 'ask':
        // Основной запрос к боту
        handle_chat_request();
        break;

    case 'login':
        // Проверка пароля администратора
        handle_admin_login();
        break;

    case 'rebuild':
        // Обновление базы знаний
        handle_rebuild_request();
        break;

    case 'clear_history':
        // Очистка истории диалога
        handle_clear_history();
        break;

    default:
        echo json_encode([
            'status' => 'error',
            'message' => 'Неизвестное действие'
        ]);
}

// Функция для обработки запроса к боту
function handle_chat_request() {
    global $bot_api_url;

    // Получаем вопрос из POST-запроса
    $question = isset($_POST['q']) ? $_POST['q'] : '';

    if (empty($question)) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Вопрос не может быть пустым',
            'answer' => 'Пожалуйста, введите ваш вопрос.'
        ]);
        return;
    }

    // Формируем URL для запроса к API бота
    $api_endpoint = $bot_api_url . '/ask';

    // Инициализируем cURL
    $ch = curl_init($api_endpoint);

    // Настраиваем параметры запроса
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, http_build_query(['q' => $question]));
    curl_setopt($ch, CURLOPT_TIMEOUT, 30); // Увеличиваем таймаут до 30 секунд

    // Получаем куки сессии из браузера и передаем их боту
    if (!empty($_COOKIE['session_id'])) {
        curl_setopt($ch, CURLOPT_COOKIE, 'session_id=' . $_COOKIE['session_id']);
    }

    // Настраиваем сохранение куки в ответе
    curl_setopt($ch, CURLOPT_HEADER, true);

    // Выполняем запрос
    $response = curl_exec($ch);

    // Проверяем на ошибки
    if ($response === false || curl_errno($ch)) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Ошибка при подключении к API бота: ' . curl_error($ch),
            'answer' => 'Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.'
        ]);
        curl_close($ch);
        return;
    }

    // Получаем код ответа HTTP
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);

    // Получаем размер заголовка и тела ответа
    $header_size = curl_getinfo($ch, CURLINFO_HEADER_SIZE);
    $header = substr($response, 0, $header_size);
    $body = substr($response, $header_size);

    curl_close($ch);

    // Проверяем HTTP код ответа
    if ($http_code !== 200) {
        echo json_encode([
            'status' => 'error',
            'message' => 'API бота вернул ошибку. Код: ' . $http_code,
            'answer' => 'Извините, сервис временно недоступен. Пожалуйста, попробуйте позже.'
        ]);
        return;
    }

    // Передаем куки из ответа бота клиенту
    preg_match_all('/^Set-Cookie:\s*([^;]*)/mi', $header, $matches);
    foreach ($matches[1] as $cookie) {
        $parts = explode('=', $cookie, 2);
        if (count($parts) === 2) {
            $name = trim($parts[0]);
            $value = trim($parts[1]);
            setcookie($name, $value, 0, '/');
        }
    }

    // Декодируем JSON из тела ответа
    $data = json_decode($body, true);

    // Проверяем успешность декодирования JSON
    if ($data === null && json_last_error() !== JSON_ERROR_NONE) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Ошибка декодирования ответа: ' . json_last_error_msg(),
            'answer' => 'Извините, произошла ошибка при обработке ответа. Пожалуйста, попробуйте позже.'
        ]);
        return;
    }

    // Возвращаем данные клиенту
    echo $body;
}

// Функция для обработки запроса на логин администратора
function handle_admin_login() {
    global $admin_password;

    // Получаем пароль из POST-запроса
    $password = isset($_POST['password']) ? $_POST['password'] : '';

    if (empty($password)) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Пароль не может быть пустым'
        ]);
        return;
    }

    // Проверяем пароль
    if ($password === $admin_password) {
        // Генерируем токен на основе хеша пароля
        $token = hash('sha256', $admin_password);

        echo json_encode([
            'status' => 'success',
            'message' => 'Авторизация успешна',
            'token' => $token
        ]);
    } else {
        echo json_encode([
            'status' => 'error',
            'message' => 'Неверный пароль администратора'
        ]);
    }
}

// Функция для обработки запроса на обновление базы знаний
function handle_rebuild_request() {
    global $bot_api_url, $admin_password;

    // Получаем токен из POST-запроса
    $token = isset($_POST['token']) ? $_POST['token'] : '';

    // Проверяем токен
    $expected_token = hash('sha256', $admin_password);
    if ($token !== $expected_token) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Доступ запрещен: недействительный токен'
        ]);
        return;
    }

    // Формируем URL для запроса к API бота
    $api_endpoint = $bot_api_url . '/rebuild';

    // Инициализируем cURL
    $ch = curl_init($api_endpoint);

    // Настраиваем параметры запроса
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 300); // Увеличиваем таймаут до 300 секунд для длительной операции
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'admin-token: ' . $token
    ]);

    // Выполняем запрос
    $response = curl_exec($ch);

    // Проверяем на ошибки
    if ($response === false || curl_errno($ch)) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Ошибка при подключении к API бота: ' . curl_error($ch)
        ]);
        curl_close($ch);
        return;
    }

    // Получаем код ответа HTTP
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);

    curl_close($ch);

    // Проверяем HTTP код ответа
    if ($http_code !== 200) {
        echo json_encode([
            'status' => 'error',
            'message' => 'API бота вернул ошибку. Код: ' . $http_code
        ]);
        return;
    }

    // Декодируем JSON из тела ответа
    $data = json_decode($response, true);

    // Проверяем успешность декодирования JSON
    if ($data === null && json_last_error() !== JSON_ERROR_NONE) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Ошибка декодирования ответа: ' . json_last_error_msg()
        ]);
        return;
    }

    // Возвращаем данные клиенту
    echo $response;
}

// Функция для обработки запроса на очистку истории диалога
function handle_clear_history() {
    global $bot_api_url;

    // Формируем URL для запроса к API бота
    $api_endpoint = $bot_api_url . '/clear-session';

    // Инициализируем cURL
    $ch = curl_init($api_endpoint);

    // Настраиваем параметры запроса
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 10);

    // Получаем куки сессии из браузера и передаем их боту
    if (!empty($_COOKIE['session_id'])) {
        curl_setopt($ch, CURLOPT_COOKIE, 'session_id=' . $_COOKIE['session_id']);
    }

    // Выполняем запрос
    $response = curl_exec($ch);

    // Проверяем на ошибки
    if ($response === false || curl_errno($ch)) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Ошибка при подключении к API бота: ' . curl_error($ch)
        ]);
        curl_close($ch);
        return;
    }

    // Получаем код ответа HTTP
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);

    curl_close($ch);

    // Проверяем HTTP код ответа
    if ($http_code !== 200) {
        echo json_encode([
            'status' => 'error',
            'message' => 'API бота вернул ошибку. Код: ' . $http_code
        ]);
        return;
    }

    // Декодируем JSON из тела ответа
    $data = json_decode($response, true);

    // Проверяем успешность декодирования JSON
    if ($data === null && json_last_error() !== JSON_ERROR_NONE) {
        echo json_encode([
            'status' => 'error',
            'message' => 'Ошибка декодирования ответа: ' . json_last_error_msg()
        ]);
        return;
    }

    // Возвращаем данные клиенту
    echo $response;
}