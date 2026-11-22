import sys
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
p = PrefixwiseProxy(mock=True); p.mock_tool_calls = True
PrefixwiseHTTPServer(("127.0.0.1", int(sys.argv[1])), make_handler(p)).serve_forever()
