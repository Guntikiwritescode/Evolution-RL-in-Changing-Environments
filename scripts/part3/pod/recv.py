# Receive exactly one file (the code package) and exit. No commands are accepted.
# PUT /<TOKEN>/code.tgz with a body whose sha256 equals $CODE_SHA256 is saved to /w/code.tgz;
# any other request, or a body with another hash, is refused. After one good upload the
# server stops, so the pod runs only the package fixed at creation time.
import hashlib, os
from http.server import HTTPServer, BaseHTTPRequestHandler

TOKEN = os.environ["TOKEN"]
WANT = os.environ["CODE_SHA256"]
os.makedirs("/w", exist_ok=True)


class H(BaseHTTPRequestHandler):
    def do_PUT(self):
        if self.path != f"/{TOKEN}/code.tgz":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        h = hashlib.sha256()
        left = n
        with open("/w/code.tgz.part", "wb") as fh:
            while left > 0:
                b = self.rfile.read(min(1 << 20, left))
                if not b:
                    break
                fh.write(b)
                h.update(b)
                left -= len(b)
        if left or h.hexdigest() != WANT:
            os.remove("/w/code.tgz.part")
            self.send_error(400, "wrong package")
            return
        os.replace("/w/code.tgz.part", "/w/code.tgz")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"received\n")
        self.server.done = True

    def do_GET(self):
        self.send_error(404)

    def log_message(self, *a):
        pass


srv = HTTPServer(("0.0.0.0", 8889), H)
srv.done = False
while not srv.done:
    srv.handle_request()
