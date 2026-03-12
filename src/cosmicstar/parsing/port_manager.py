import socket
from contextlib import closing

class PortManager:
    """
    Manages allocation of ports for SC2 instances.
    Thread-safe if used carefully (checking for open ports).
    """
    
    def __init__(self, start_port=12000):
        self._next_port = start_port

    def find_free_port(self) -> int:
        """Finds a single free port."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            return port

    def get_port_set(self, count=2) -> list[int]:
        """
        Get a set of sequential or random free ports.
        SC2 often needs a block (game port + base port per player).
        For single player replay, we just need enough for the server.
        """
        ports = []
        for _ in range(count):
             # This is a naive implementation; for high concurrency 
             # we might want to actually bind temporary sockets to reserve them.
             # But for this task, finding free ports is usually sufficient.
             ports.append(self.find_free_port())
        return ports
