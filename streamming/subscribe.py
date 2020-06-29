import zmq
import time
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.setsockopt_string(zmq.IDENTITY, "Hello")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
subscriber.connect("tcp://192.168.0.58:5565")

sync = context.socket(zmq.PUSH)
sync.connect("tcp://192.168.0.58:5564")
sync.send_string("")
while True:
    data = subscriber.recv()
    print(data)
    if data =="END":
        break


