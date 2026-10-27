"""Random access to files inside a split zip archive (name.z01, name.z02, ..., name.zip).

The UniqueShip audio comes as a 2-part split zip (10 GB + 6.6 GB). Instead of joining
and unzipping 27.7 GB, we read the archive's table of contents (the "central directory",
stored at the end of the last part) once, and then jump straight to any clip we need.

    z = SplitZip(r"D:\\shipsound\\gdrive_download\\audio\\allclass_5hours.zip")
    wav_bytes = z.read("allclass_5hours/audio/tug/20170511T112930.000Z_20170511T112935.000Z.wav")
"""
import os
import struct
import threading
import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    name: str
    method: int        # 0 = stored, 8 = deflate
    crc: int
    csize: int         # compressed size
    usize: int         # uncompressed size
    disk: int          # which part the file starts in (0 = .z01, last = .zip)
    offset: int        # offset of the local header inside that part


class SplitZip:
    def __init__(self, last_part):
        base = last_part[:-4]
        parts, i = [], 1
        while os.path.exists(f"{base}.z{i:02d}"):
            parts.append(f"{base}.z{i:02d}")
            i += 1
        self.paths = parts + [last_part]
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self._local = threading.local()          # one set of file handles per thread
        self.entries = {e.name: e for e in self._read_central_directory()}

    # ---------- low level ----------
    def _handles(self):
        if not hasattr(self._local, "fh"):
            self._local.fh = [open(p, "rb") for p in self.paths]
        return self._local.fh

    def _read(self, disk, off, n):
        """Read n bytes starting at (disk, off), continuing into the next part if needed."""
        fh, out = self._handles(), bytearray()
        while n > 0:
            if off >= self.sizes[disk]:
                off -= self.sizes[disk]
                disk += 1
                continue
            fh[disk].seek(off)
            chunk = fh[disk].read(min(n, self.sizes[disk] - off))
            out += chunk
            n -= len(chunk)
            off += len(chunk)
        return bytes(out), disk, off

    def _read_central_directory(self):
        last = len(self.paths) - 1
        tail_len = min(self.sizes[last], 1 << 20)
        tail, _, _ = self._read(last, self.sizes[last] - tail_len, tail_len)
        e = tail.rfind(b"PK\x05\x06")                                 # end-of-central-directory record
        _, _, cd_disk, _, n, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[e:e + 22])
        if 0xFFFF in (cd_disk, n) or 0xFFFFFFFF in (cd_size, cd_off):  # zip64 archive: real values elsewhere
            loc = tail.rfind(b"PK\x06\x07", 0, e)
            _, z64_disk, z64_off, _ = struct.unpack("<IIQI", tail[loc:loc + 20])
            rec, _, _ = self._read(z64_disk, z64_off, 56)
            _, _, _, _, _, cd_disk, _, n, cd_size, cd_off = struct.unpack("<IQHHIIQQQQ", rec)

        cd, _, _ = self._read(cd_disk, cd_off, cd_size)
        p, entries = 0, []
        for _ in range(n):
            (sig, _, _, _, method, _, _, crc, csize, usize, fnl, exl, cml,
             disk, _, _, off) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
            if sig != 0x02014B50:
                raise ValueError("corrupt central directory")
            name = cd[p + 46:p + 46 + fnl].decode("utf-8")
            extra = cd[p + 46 + fnl:p + 46 + fnl + exl]
            q = 0
            while q + 4 <= len(extra):                                # zip64 extra field holds big sizes/offsets
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    v, r = extra[q + 4:q + 4 + hsz], 0
                    if usize == 0xFFFFFFFF: usize = struct.unpack("<Q", v[r:r + 8])[0]; r += 8
                    if csize == 0xFFFFFFFF: csize = struct.unpack("<Q", v[r:r + 8])[0]; r += 8
                    if off == 0xFFFFFFFF: off = struct.unpack("<Q", v[r:r + 8])[0]; r += 8
                    if disk == 0xFFFF: disk = struct.unpack("<I", v[r:r + 4])[0]
                q += 4 + hsz
            if not name.endswith("/"):
                entries.append(Entry(name, method, crc, csize, usize, disk, off))
            p += 46 + fnl + exl + cml
        return entries

    # ---------- public ----------
    def names(self):
        return list(self.entries)

    def read(self, name, check_crc=True):
        e = self.entries[name]
        hdr, _, _ = self._read(e.disk, e.offset, 30)
        fnl, exl = struct.unpack("<HH", hdr[26:30])
        _, disk, off = self._read(e.disk, e.offset, 30 + fnl + exl)
        raw, _, _ = self._read(disk, off, e.csize)
        if e.method == 0:
            data = raw
        elif e.method == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise NotImplementedError(f"compression method {e.method} for {name}")
        if check_crc and zlib.crc32(data) != e.crc:
            raise IOError(f"CRC mismatch in {name}")
        return data

    def sort_key(self, name):
        """Position in the archive. Reading in this order is fastest (sequential disk access)."""
        e = self.entries[name]
        return (e.disk, e.offset)
