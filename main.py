import pika, time, json, os, sys, dotenv
from service.rabbitmq import RabbitMQ
from service.ai_service_processor import AIProcessor
from langchain_core.output_parsers import JsonOutputParser
from models.user import UserDataForGuidence, AIGuidanceResponse
from utils.utils import init_mongo
from config import settings as global_settings

dotenv.load_dotenv()
        
def main():
    print("testing")
    rabbitmq = RabbitMQ()
    def callback(ch, method, properties, body):
       print(f"{ch}, {body}, {properties}")

    rabbitmq.channel.queue_declare("ai_tasks")
    rabbitmq.channel.queue_declare("ai_response")

    print(rabbitmq.__dict__)
    
    try:
        print("Connection to RabbitMQ established successfully.")
        rabbitmq.consume(queue_name='ai_tasks', callback=callback)
        
    except Exception as e:
        print(f"Failed to establish connection to RabbitMQ: {e}")
        sys.exit(1)
    finally:
        rabbitmq.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)