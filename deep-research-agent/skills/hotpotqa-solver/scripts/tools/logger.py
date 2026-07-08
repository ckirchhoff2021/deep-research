"""日志工具"""
import os
import logging


def setup_logger(name: str, level: int = logging.INFO, log_file: str = None, stream: bool = True) -> logging.Logger:
    """创建并配置 logger
    
    Args:
        name: logger 名称
        level: 日志级别
        log_file: 日志文件路径，若提供则写入文件
        stream: 是否输出到终端，默认 True
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # 终端输出
        if stream:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)
        
        # 文件输出
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        logger.setLevel(level)
    return logger
