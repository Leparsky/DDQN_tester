#http://spyne.io/#inprot=HttpRpc&outprot=JsonDocument&s=rpc&tpt=TwistedWebResource&validator=true
import logging
from environment import Environment
from DDQN.agent import Agent
import sys
logging.basicConfig(level=logging.DEBUG)
from spyne import Application, rpc, ServiceBase, \
    Integer, Unicode
from spyne import Iterable
from spyne.protocol.http import HttpRpc
from spyne.protocol.json import JsonDocument
from spyne.server.wsgi import WsgiApplication
from main import parse_args


class HelloWorldService(ServiceBase):
    @rpc(Unicode, Integer, _returns=Iterable(Unicode))
    def say_hello(ctx, name, times):
        yield "Hi friend"



application = Application([HelloWorldService],
                          tns='spyne.examples.hello',
                          in_protocol=HttpRpc(validator='soft'),
                          out_protocol=JsonDocument()
                          )

def serve(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    env = Environment(args)
    env.GetStockDataVecFN("SomeData.csv", False)
    env.check_data_integrity4prediction_at("2018-09-03 10:05:00")
    agent = Agent(env.get_state_size(), env.get_action_size(), 2.5e-4 #lr
          , 1e-2 #env.tau
          , args.dueling, args.hidden_dim)
    agent.loadModel(r"D:\PycharmProjects\DDQN_tester\model_1")

    from wsgiref.simple_server import make_server

    wsgi_app = WsgiApplication(application)
    server = make_server('0.0.0.0', 8000, wsgi_app)
    server.serve_forever()

if __name__ == '__main__':
    serve()