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

import socket, ssl, sys, multiprocessing, pickle
from starttf.cluster.gpu_checker import is_gpu_free

training = None

def start_training(cmd, conn):
  try:
    obj = pickle.loads(cmd)
    fun, params = obj
    fun(*params)
  except Exception, exception:
    print(exception)
    traceback.print_exc()
  global training
  training = None
  print("DONE TRAINING")

def handle(conn, password_list):
  fh = conn.makefile()
  handshake = fh.readline().decode("utf-8")
  if handshake.startswith("handshake ") and handshake[len("handshake "):] in password_list:
    pass
  else:
    conn.write("Handshake failed!\n")
    conn.close()
    return
  while True:
    cmd = fh.readline().decode("utf-8")
    msg = "Hello World!"

    global training
    if cmd.startswith('start '):
      if is_gpu_free() and training is None:
        sys.stdout = conn.makefile('w')
        sys.stderr = conn.makefile('w')
        training = multiprocessing.Process(target=start_training, args=(cmd[len("start "):],))
        training.start()
        msg = "started training"
      else:
        msg = "ERROR GPU occupied."
    elif cmd.startswith('kill'):
      if training is None:
        msg = "ERROR No training running."
      else:
        training.terminate()
        training = None
        msg = "killed training"
    elif cmd.startswith('reconnect'):
      if training is None:
        msg = "ERROR No training running."
      else:
        sys.stdout = conn.makefile('w')
        sys.stderr = conn.makefile('w')
        msg = "reconnected training"
    else:
      return
    conn.write(msg.encode("utf-8"))

def main(host, port, certs_folder='.', password_list=[]):
  cert = certs_folder + '/' + host + '.pem'
  sock = socket.socket()
  sock.bind((host, port))
  sock.listen(5)
  context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
  context.load_cert_chain(certfile=cert)  # 1. key, 2. cert, 3. intermediates
  context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # optional
  context.set_ciphers('EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH')
  while True:
    conn = None
    ssock, addr = sock.accept()
    try:
      conn = context.wrap_socket(ssock, server_side=True)
      handle(conn, password_list)
    except ssl.SSLError as e:
      print(e)
    finally:
      if conn:
        conn.close()


if __name__ == '__main__':
  import sys
  main(sys.argv[1], int(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else '.'))
