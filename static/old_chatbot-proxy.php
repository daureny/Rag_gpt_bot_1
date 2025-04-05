<?php
// Простой прокси для вашего локального чат-бота
// Сохраните этот файл как chatbot-proxy.php в корне сайта

// Разрешаем запросы только со своего сайта
header('Access-Control-Allow-Origin: ' . $_SERVER['HTTP_ORIGIN']);
header('Access-Control-Allow-Methods: POST');
header('Access-Control-Allow-Headers: Content-Type');

// Получаем вопрос от пользователя
$question = isset($_POST['q']) ? $_POST['q'] : '';

if (empty($question)) {
    echo json_encode(['answer' => 'Ошибка: пустой запрос']);
    exit;
}

// URL вашего локального бота
$bot_url = 'https://2f93-2a03-32c0-2d-d051-716a-650e-df98-8a9f.ngrok-free.app/ask'; // Заменил на локальный URL

// Отправляем запрос к локальному боту
$ch = curl_init($bot_url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, http_build_query(['q' => $question]));
$response = curl_exec($ch);
curl_close($ch);

// Возвращаем ответ от бота
echo $response;
