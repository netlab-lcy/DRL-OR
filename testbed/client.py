#coding=utf-8
import socket
import sys
import time

BUFFER_SIZE = 128

def padding_bytes(x, target_len):
    clen = len(x)
    x += bytes(target_len - clen)
    return x

if __name__ == '__main__':
    server_addr = sys.argv[1]
    server_port = int(sys.argv[2])
    client_addr = sys.argv[3]
    client_port = int(sys.argv[4])
    demand = int(sys.argv[5]) # Kbps
    rtime = int(sys.argv[6]) # seconds

    time_step = int(sys.argv[7]) #for testing 
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((client_addr, client_port))
    ind = 0
    start_time = time.time()
    time_stamp = int(time.time() * 1000)
    while True:
        temp_stamp = time.time()
        msg = "%d;%d;" % (ind, int(temp_stamp * 1000))
        msg = padding_bytes(msg.encode(), BUFFER_SIZE)
        sock.sendto(msg, (server_addr, server_port))
        
        ind += 1
        curr_bit = ind * BUFFER_SIZE * 8
        temp_stamp = time.time()
        if curr_bit > (temp_stamp - start_time) * demand * 1000:
            time.sleep(BUFFER_SIZE / (demand * 125))
    sock.close()
