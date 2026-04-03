from dataclasses import dataclass
import multiprocessing as mp


@dataclass
class Document:
    doc: str = ""
    doc_id: int = 0
    num_chunks: int = 0

class ProtocolConstants:

    @staticmethod
    def expect(q: mp.Queue, constant):
        k, v = q.get()
        assert k == constant, f"expect {constant} but got {k}"
        return v
    
    @staticmethod
    def send(q: mp.Queue, constant, data=None, block=True):
        q.put((constant, data), block=block)
