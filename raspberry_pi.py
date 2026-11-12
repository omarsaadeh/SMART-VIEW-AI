import socket
import subprocess

PORT = 5014

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Allow immediate re-binding to port upon script restarts without address conflicts
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", PORT))
s.listen(1)

print("Raspberry Pi Audio Node online. Waiting for connection...")

try:
    while True:
        conn, addr = s.accept()
        print("Connected safely to Processing Core at:", addr)

        # Maintain a residual buffer to handle fragmented packets over TCP socket
        buffer = ""

        while True:
            data = conn.recv(1024)
            if not data:
                print("Processing Core disconnected.")
                break

            # Append newly arrived decoded characters into stream buffer
            buffer += data.decode('utf-8', errors='ignore')

            # Extract out individual phrases terminated by the newline character (\n)
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                label = line.strip()

                if label:
                    print("Speaking output text:",  label)
                    # -s 160 alters speed, adjustments can be applied straight onto terminal execution call
                    subprocess.run(["espeak", "-s", "150","-g","10",  label])

        conn.close()
except KeyboardInterrupt:
    print("\nShutting down Audio Server node.")
finally:
    s.close()
