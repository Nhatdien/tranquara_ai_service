import pika
import os
import time


class RabbitMQ:
    def __init__(self):
        self.user = os.getenv('RABBITMQ_USER', 'guest')
        self.password = os.getenv('RABBITMQ_PASSWORD', 'guest')
        self.host = os.getenv('RABBITMQ_HOST', 'rabbitmq')
        self.port = int(os.getenv('RABBITMQ_PORT', 5672))
        self.connection = None
        self.channel = None

        self.connect()

    def connect(self):

        print("calling conenct")
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host, port=self.port, credentials=credentials, blocked_connection_timeout=200, connection_attempts=5)
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def consume(self, queue_name, callback):
        if not self.channel:
            raise Exception("Connection is not established.")
        self.channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True)
        self.channel.start_consuming()

    def publish(self, queue_name, message):
        if not self.channel:
            raise Exception("Connection is not established.")
        self.channel.basic_publish(exchange='',
                                   routing_key=queue_name,
                                   body=message,
                                   properties=pika.BasicProperties(
                                       delivery_mode=2,  # make message persistent
                                   ))
        print(f"Sent message to queue {queue_name}: {message}")


rabbitmq_conn = RabbitMQ()
