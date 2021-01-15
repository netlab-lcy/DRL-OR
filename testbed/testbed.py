from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import Link, Intf, TCLink
from mininet.topo import Topo
from mininet.util import custom, pmonitor
import logging
import os
from functools import partial
import socket
import json
import time
import heapq
import sys

class CustomTopo(Topo):
    def __init__(self, nodeNum, linkSet, bandwidths, losses, **opts):
        Topo.__init__(self,**opts)
        self.__nodenum = nodeNum
        self.__linkset = linkSet
        self.__bandwidths = bandwidths
        self.__losses = losses

        self.__switches = []
        self.__hosts = []

        self.create_net()
        self.add_hosts()

    '''create the network topo'''
    def create_net(self):
        for i in range(self.__nodenum):
            self.__switches.append(self.addSwitch("s" + str(i + 1)))
        for i in range(len(self.__linkset)):
            node1 = self.__linkset[i][0]
            node2 = self.__linkset[i][1]
            self.addLink(self.__switches[node1], self.__switches[node2], bw=self.__bandwidths[i], delay='5ms', loss=self.__losses[i], max_queue_size=1000) 
    
    '''add host for each switch(node)'''
    def add_hosts(self):
        if self.__nodenum >= 255:
            print("ERROR!!!")
            exit()
        for i in range(self.__nodenum):
            self.__hosts.append(self.addHost("h" + str(i + 1), mac=("00:00:00:00:00:%02x" % (i + 1)), ip = "10.0.0." + str(i + 1)))
            self.addLink(self.__switches[i], self.__hosts[i], bw=1000, delay='0ms') # bw here should be large enough
        

def generate_request(net, src, src_port, dst, dst_port, rtype, demand, rtime, time_step): 
    TIME_OUT = 5
    src_host = net.hosts[src]
    dst_host = net.hosts[dst]

    popens = {}
    popens[dst_host] = dst_host.popen("python3 server.py %s %d %d %d %d" % (dst_host.IP(), dst_port, rtime, rtype, time_step))
    time.sleep(0.1)
    popens[src_host] = src_host.popen("python3 client.py %s %d %s %d %d %d %d" % (dst_host.IP(), dst_port, src_host.IP(), src_port, demand, rtime, time_step))
    src_popen = popens[src_host]
    dst_popen = popens[dst_host]
    ind = 0
    time_stamp = time.time()
    for host, line in pmonitor(popens):
        if time.time() - time_stamp > TIME_OUT:
            print("Request:", "src:", src, "dst:", dst, "rtype:", rtype, "demand:", demand)
            delay = TIME_OUT * 1000
            throughput = 0
            loss = 1.
            print("time out!")
            break
        if host:
            print("<%s>: %s" % (host.name, line))
            
            if host == dst_host:
                ret = line.split()
                delay = float(ret[1])
                throughput = float(ret[4])
                loss = float(ret[7])
                #flag = True
                if ind == 1: # avoid using the first data received from server
                    break
                else:
                    ind += 1
            
    return delay, throughput, loss, (src_popen, dst_popen)

def load_topoinfo(toponame):
    topo_file = open("./topo_info/%s.txt" % toponame, "r")
    content = topo_file.readlines()
    nodeNum, linkNum = map(int, content[0].split())
    linkSet = []
    bandwidths = []
    losses = []
    for i in range(linkNum):
        u, v, w, c, loss = map(int, content[i + 1].split())
        linkSet.append([u - 1, v - 1])
        bandwidths.append(float(c) / 1000) 
        losses.append(loss)
    return nodeNum, linkSet, bandwidths, losses

if __name__ == '__main__':
    print "testbed initializing ..."
    toponame = sys.argv[1]
    if toponame == "test":
        nodeNum = 4
        linkSet = [[0, 1], [1, 2], [2, 3], [0, 3]]
        bandwidths = [1, 5, 5, 5]
        losses = [0, 0, 0, 0] # 0% must be int
    else:
        nodeNum, linkSet, bandwidths, losses = load_topoinfo(toponame)
    print "topoinfo loading finished."
    requests_pq = [] # put the popens of requests' server and client process
    
    topo = CustomTopo(nodeNum, linkSet, bandwidths, losses)
    CONTROLLER_IP = "127.0.0.1" # Your ryu controller server IP
    CONTROLLER_PORT = 5001 
    OVSSwitch13 = partial(OVSSwitch, protocols='OpenFlow13')
    net = Mininet(topo=topo, switch=OVSSwitch13, link=TCLink, controller=None)
    net.addController('controller', controller=RemoteController, ip=CONTROLLER_IP, port=CONTROLLER_PORT)
    net.start()
    
    
    # build communication with DRL client
    print "waiting to simenv"
    # If using unique server for testbed, set TCP_IP to the server IP 
    TCP_IP = "127.0.0.1"
    TCP_PORT = 5000
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    
    conn, addr = s.accept()
    print('Connection address:', addr)
    time_step = 0
    # receive instruction from sim_env.py and generate request and send results
    while True:
        try:
            msg = conn.recv(BUFFER_SIZE)
        except:
            sock.close()
            break
        #print("msg:", msg)
         
        while len(requests_pq) > 0 and requests_pq[0][0] <= time_step:
            ind, popens = heapq.heappop(requests_pq)
            popens[0].kill()
            popens[1].kill()
        

        data_js = json.loads(msg)
        rtime = data_js['rtime']
        delay, throughput, loss, popens = generate_request(net, data_js['src'], data_js['src_port'], data_js['dst'], data_js['dst_port'], data_js['rtype'], data_js['demand'], 1000000, time_step) # rtime is a deprecated para
        
        heapq.heappush(requests_pq, (rtime + time_step, popens))
        
        ret = {
                'delay': delay,
                'throughput': throughput,
                'loss': loss,
                }
        
        # For Abi link failure & demand change test
        # we let testbed send the failure information to simenv for simple implementation 
        if time_step == 10000:
            '''
            # link failue
            net.configLinkStatus('s1', 's5', 'down')
            ret['change'] = 'link_failure'
            '''
            # demand change
            #ret['change'] = "demand_change"
        
        
        msg = json.dumps(ret)
        conn.send(msg.encode())
        time_step += 1

    CLI(net)
    

