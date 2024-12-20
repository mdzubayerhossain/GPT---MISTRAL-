CREATE DATABASE IF NOT EXISTS `user-system`;
USE `user-system`;

CREATE TABLE conversation_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    response TEXT NOT NULL
);
