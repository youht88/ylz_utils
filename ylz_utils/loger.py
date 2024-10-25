import logging
from logging.handlers import TimedRotatingFileHandler

class LoggerLib():
    def init(project_name:str,log_level:str=logging.INFO):
        # 设置logger
        log_file = f"{project_name}.log"
        logging.basicConfig(level=log_level)
        
        file_handler = TimedRotatingFileHandler(
            filename= log_file,
            when="midnight",  # 每天午夜滚动
            interval=1,  # 滚动间隔为 1 天
            backupCount=7,  # 保留 7 天的日志文件
            encoding='utf-8'
        )

        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(log_level)
        
        #file_handler = logging.FileHandler("task.log")
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s %(name)s [pid:%(process)d] [%(threadName)s] [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        #console_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
        #logger.addHandler(console_handler)
