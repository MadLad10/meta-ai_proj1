"""Read single files from DUO's official Figshare zip with HTTP range requests (no full 3.4 GB download)."""
import io, urllib.error, urllib.request, zipfile

DUO_URL = 'https://ndownloader.figshare.com/files/45670494'   # DUO.zip, Figshare record 25370527 (CC BY 4.0)


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.origin = url; self.pos = 0; self._resolve()
    def _resolve(self):                                      # the signed download URL expires after a few minutes
        req = urllib.request.Request(self.origin, method='HEAD', headers={'User-Agent': 'duo-figure/0.1'})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl(); self.size = int(r.headers['Content-Length'])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b''
        end = min(self.size, self.pos + n) - 1
        for attempt in range(3):
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}', 'User-Agent': 'duo-figure/0.1'})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 403 or attempt == 2:
                    raise
                self._resolve()
        self.pos += len(data); return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def open_duo():
    return zipfile.ZipFile(io.BufferedReader(RangeFile(DUO_URL), buffer_size=1 << 20))
