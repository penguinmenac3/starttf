# MIT License
# 
# Copyright (c) 2018 Michael Fuerst
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import socket, ssl

def handle(conn):
    fh = conn.makefile()
    while True:
        msg = fh.readline().decode("utf-8")
        if msg == "DONE TRAINING":
            return True
        print(msg)
        if msg.startswith("ERROR"):
            return False
    return True

def send_remote(command, fun, params, host, port, password):
    ret_val = False
    sock = socket.socket(socket.AF_INET)
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # optional
    conn = context.wrap_socket(sock, server_hostname=host)
    try:
        conn.connect((host, port))
        handshake = "handshake " + password + "\n"
        conn.write(handshake.encode("utf-8"))
        if command == "start":
            msg = "start " + pickle.dumps((fun, params)) + "\n"
            conn.write(msg.encode("utf-8"))
        else:
            msg = command + "\n"
            conn.write(msg.encode("utf-8"))
        ret_val = handle(conn)
    finally:
        conn.close()

    return ret_val
