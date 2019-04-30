import torch


def verbose_cuda():    
    if torch.cuda.is_available():
        dev_no = torch.cuda.device_count()
        print("GPU details:")
        print("---------------")
        print("Available devices: ", dev_no)
        for d in range(dev_no):
            print(torch.cuda.get_device_name(d))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(d)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(d)/1024**3,1), 'GB')
            print("---------------")


def convert_time(start, stop): 
    total_seconds = stop-start
    seconds = total_seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60      
    
    return round(hour), round(minutes), round(seconds)