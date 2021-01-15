# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, arp
from ryu.lib.packet import ether_types
from ryu.lib import hub
from ryu import cfg
import json
import socket

def load_topoinfo(toponame):
    topo_file = open("../drl-or/net_env/inputs/%s/%s.txt" % (toponame, toponame))
    content = topo_file.readlines()
    nodeNum, linkNum = list(map(int, content[0].split()))
    linkSet = []
    for i in range(linkNum):
        u, v, w, c, loss = list(map(int, content[i + 1].split()))
        linkSet.append([u - 1, v - 1])
    return nodeNum, linkSet

class Controller(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Controller, self).__init__(*args, **kwargs)
        CONF = cfg.CONF
        CONF.register_opts([
            cfg.StrOpt("toponame", default="test", help=("network topology name"))])

        self.toponame = CONF.toponame

        if self.toponame == "test":
            self.node_num = 4
            self.linkset = [[0, 1], [1, 2], [2, 3], [0, 3]]
        else:
            self.node_num, self.linkset = load_topoinfo(self.toponame)
        print("loading topoinfo finished")
        # preset topo physic port info, not being used now
        # switch indexed from 1(dpid) while node indexed from 0
        self.link_port = {} # indexed by dpid
        for i in range(self.node_num):
            self.link_port[i + 1] = {}
        switch_port_counter = [1] * self.node_num
        for link in self.linkset:
            u, v = link
            self.link_port[u + 1][v + 1] = switch_port_counter[u]
            self.link_port[v + 1][u + 1] = switch_port_counter[v]
            switch_port_counter[u] += 1
            switch_port_counter[v] += 1
        self.switch_host_port = {}
        for i in range(self.node_num):
            self.switch_host_port[i + 1] = switch_port_counter[i] 
        
        self.action_rule_priority = 5

        self.datapaths = {} # dpid to datapaths, indexed from 1
        
        self.install_rules_thread = hub.spawn(self.install_rules)
    
    '''Install statics rules and dynamically install rules for new requests '''
    def install_rules(self):
        
        TCP_IP = "127.0.0.1" 
        TCP_PORT = 3999
        BUFFER_SIZE = 1024
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)

        self.simenv_socket, addr = s.accept()
        print("Connection address:", addr)

        while len(self.datapaths) < self.node_num:
            hub.sleep(5)

        print("Ready to install dynamic rules for requests.")
        while True:
            msg = self.simenv_socket.recv(BUFFER_SIZE)
            data_js = json.loads(msg.decode('utf-8'))
            # install rules by (src_ip, src_port, dst_ip, dst_port, protocol_type)
            path = data_js['path']
            src_port = data_js['src_port']
            dst_port = data_js['dst_port']
            ipv4_src = data_js['ipv4_src']
            ipv4_dst = data_js['ipv4_dst']
            print("path:", path)
            
            temp = {}
            for i in range(len(path)):
                temp[path[i]] = 1
                
                dpid = path[i] + 1
                datapath = self.datapaths[dpid]
                ofproto = datapath.ofproto
                parser = datapath.ofproto_parser
                match = parser.OFPMatch(ipv4_src=ipv4_src, udp_src=src_port, ipv4_dst=ipv4_dst, udp_dst=dst_port, ip_proto=17, eth_type=0x0800)
                
                # delete the ring part of 
                if i < len(path) - 1 and path[i + 1] in temp:
                    actions = []
                    self.add_flow(datapath, self.action_rule_priority, match, actions)
                    break  
                
                if i == len(path) - 1:
                    out_port = self.switch_host_port[dpid]
                else:
                    out_port = self.link_port[dpid][path[i + 1] + 1]
                
                actions = [parser.OFPActionOutput(out_port)]
                self.add_flow(datapath, self.action_rule_priority, match, actions)
                

            self.simenv_socket.send("Succeeded!".encode())
                


    '''Set up when the switches connected to Controller'''
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        dpid = datapath.id
        self.datapaths[dpid] = datapath
        # install table-miss flow entry
        #
        # We specify NO BUFFER to max_len of the output action due to
        # OVS bug. At this moment, if we specify a less number, e.g.,
        # 128, OVS will send Packet-In with invalid buffer_id and
        # truncated packet data. In that case, we cannot output packets
        # correctly.  The bug has been fixed in OVS v2.1.0.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
    
    '''Function to add a flow rule, the larger the prioity is, the more important'''
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)
    
    '''
        Handle the packet that not matched in the current rules
        We handle arp and other unknown packets(throw away) here.
    '''
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)

        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # handle arp request(and answer)
        arppkt = pkt.get_protocol(arp.arp)
        if arppkt != None:
            for nodeid in range(self.node_num):
                #send packet to all of the host by controller
                actions = [parser.OFPActionOutput(self.switch_host_port[nodeid + 1])] # indexed from 1
                datapath = self.datapaths[nodeid + 1]
                out = parser.OFPPacketOut(
                    datapath=datapath,
                    buffer_id=ofproto.OFP_NO_BUFFER,
                    in_port=ofproto.OFPP_CONTROLLER,
                    actions=actions, data=msg.data)
                datapath.send_msg(out)
            return
        
        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)



    
