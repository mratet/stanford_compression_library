import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
from fontTools.ttLib.sfnt import sfntDirectoryFormat
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import fftpack  # TODO, a explorer
from dataclasses import dataclass
import struct
from bitstring import *


# Start of Frame markers, non-differential, Huffman coding
SOF0 = 0xC0  # Baseline DCT
SOF1 = 0xC1  # Extended sequential DCT
SOF2 = 0xC2  # Progressive DCT
SOF3 = 0xC3  # Lossless (sequential)
# Start of Frame markers, differential, Huffman coding
SOF5 = 0xC5  # Differential sequential DCT
SOF6 = 0xC6  # Differential progressive DCT
SOF7 = 0xC7  # Differential lossless (sequential)
# Start of Frame markers, non-differential, arithmetic coding
SOF9 = 0xC9  # Extended sequential DCT
SOF10 = 0xCA  # Progressive DCT
SOF11 = 0xCB  # Lossless (sequential)
# Start of Frame markers, differential, arithmetic coding
SOF13 = 0xCD  # Differential sequential DCT
SOF14 = 0xCE  # Differential progressive DCT
SOF15 = 0xCF  # Differential lossless (sequential)
# Define Huffman Table(s)
DHT = 0xC4
# JPEG extensions
JPG = 0xC8
# Define Arithmetic Coding Conditioning(s)
DAC = 0xCC
# Restart interval Markers
RST0 = 0xD0
RST1 = 0xD1
RST2 = 0xD2
RST3 = 0xD3
RST4 = 0xD4
RST5 = 0xD5
RST6 = 0xD6
RST7 = 0xD7
# Other Markers
SOI = 0xD8  # Start of Image
EOI = 0xD9  # End of Image
SOS = 0xDA  # Start of Scan
DQT = 0xDB  # Define Quantization Table(s)
DNL = 0xDC  # Define Number of Lines
DRI = 0xDD  # Define Restart Interval
DHP = 0xDE  # Define Hierarchical Progression
EXP = 0xDF  # Expand Reference Component(s)
# APPN Markers
APP0 = 0xE0
APP1 = 0xE1
APP2 = 0xE2
APP3 = 0xE3
APP4 = 0xE4
APP5 = 0xE5
APP6 = 0xE6
APP7 = 0xE7
APP8 = 0xE8
APP9 = 0xE9
APP10 = 0xEA
APP11 = 0xEB
APP12 = 0xEC
APP13 = 0xED
APP14 = 0xEE
APP15 = 0xEF
# Misc Markers
JPG0 = 0xF0
JPG1 = 0xF1
JPG2 = 0xF2
JPG3 = 0xF3
JPG4 = 0xF4
JPG5 = 0xF5
JPG6 = 0xF6
JPG7 = 0xF7
JPG8 = 0xF8
JPG9 = 0xF9
JPG10 = 0xFA
JPG11 = 0xFB
JPG12 = 0xFC
JPG13 = 0xFD
COM = 0xFE
TEM = 0x01


def zigzag(n):
    """zigzag rows"""

    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)

    xs = range(n)
    zigzag = {
        n: index
        for n, index in enumerate(sorted(((x, y) for x in xs for y in xs), key=compare))
    }
    return {id: 8 * x + y for id, (x, y) in zigzag.items()}


def get_zigzag_order(block_size=8):
    return zigzag(block_size)

@dataclass
class QuantizationTable:
    table: tuple  # uint 64
    set: bool = False


@dataclass
class HuffmanTable:
    offsets: tuple  # byte 17
    symbols: tuple  # byte 162
    codes: list
    set: bool = False


@dataclass
class ColorComponent:  # byte pour chaque
    horizontalSamplingFactor: int
    verticalSamplingFactor: int
    quantizationTableID: int
    huffmanDCTableID: int = -1
    huffmanACTableID: int = -1
    used: bool = False


class JPEGReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.header = {
            "QuantizationTables": [  0, 0, 0, 0  ],
            "HuffmanDCTables": [  0, 0, 0, 0  ],
            "HuffmanACTables": [  0, 0, 0, 0  ],
            "frameType": 0,
            "height": 0,
            "width": 0,
            "numComponents": 0,
            "startOfSelection": False,
            "endOfSelection": False,
            "successiveApproximationHigh": False,
            "successiveApproximationLow": False,
            "restartInterval": 0,
            "ColorComponents": [  0, 0, 0  ],
            "huffmanData": [],
            "zeroBased": False,
            "valid": True}
        self.zigzagMap = get_zigzag_order()
        self.reverseZigzagMap = {v: k for k, v in self.zigzagMap.items()}

    def _uint(self, bytes):
        (uint,) = struct.unpack(">H", bytes)
        return uint

    def split_byte(self, byte):
        (byte,) = struct.unpack("B", byte)
        # table_info, = f.read(1)
        # table_precision = table_info & (1 << 5) # Bool on the first half bytes
        # tableID = table_info & 15  # read second half
        return byte // 16, byte % 16

    def readStartOfFrame(self, f):
        print("Reading Start of Frame")
        assert self.header["numComponents"] == 0 # We should find an unique FrameMarker
        L = self._uint(f.read(2))
        precision, = f.read(1)
        assert precision == 8 # JPEG does not support other options than 8 bits for RGB
        self.header["height"] = self._uint(f.read(2))
        self.header["width"] = self._uint(f.read(2))
        numComponents, = f.read(1) # GrayScale JPEG or YCbCr JPEG
        assert 1 <= numComponents <= 3 # CMYK not supported
        for i in range(numComponents):
            colorComponentID, = f.read(1)  # id, generalement 1, 2, 3. Parfois 0, 1, 2
            if colorComponentID == 0:
                self.header["zeroBased"] = True
            if self.header["zeroBased"]:
                colorComponentID += 1

            assert colorComponentID != 4 and colorComponentID != 5 # YIQ color mode not supported
            assert colorComponentID != 0 and colorComponentID <= 3 # Invalid component

            horizontalSamplingFactor, verticalSamplingFactor = self.split_byte(
                f.read(1)
            )
            # Only 1 and 1 is currently supported
            quantizationTableID, = f.read(1)
            assert quantizationTableID <= 3 # Invalid quantization table ID in frame components
            self.header["ColorComponents"][colorComponentID - 1] = ColorComponent(
                horizontalSamplingFactor, verticalSamplingFactor, quantizationTableID
            )
            assert not self.header["ColorComponents"][colorComponentID - 1].used
            self.header["ColorComponents"][colorComponentID - 1].used = True
        assert L == 8 + (3 * numComponents)
        self.header["numComponents"] = numComponents
        return

    def readQuantizationTable(self, f):
        print("Reading Quantization Tables")
        L = self._uint(f.read(2))
        L -= 2
        while L > 0:
            table_precision, tableID = self.split_byte(f.read(1))
            L -= 1
            if tableID > 3:
                raise RuntimeError("Not a valid quantization table")
            if table_precision:
                raw_bytes = struct.unpack(">64H", f.read(128))
                quant_table = QuantizationTable(tuple([raw_bytes[self.reverseZigzagMap[i]] for i in range(64)]), set = True)
                self.header["QuantizationTables"][tableID] = quant_table
                L -= 128
            else:
                raw_bytes = struct.unpack("64B", f.read(64))
                quant_table = QuantizationTable(tuple([raw_bytes[self.reverseZigzagMap[i]] for i in range(64)]), set = True)
                self.header["QuantizationTables"][tableID] = quant_table
                L -= 64

        assert L == 0
        return

    def readHuffmanTable(self, f):
        print("Reading Huffman Tables")
        offsets = [0] * 17
        symbols = [0] * 162
        L = self._uint(f.read(2))
        L -= 2
        while L > 0:
            ACTable, tableID = self.split_byte(f.read(1))
            # L -= 1 not counted ???
            assert tableID <= 3

            allSymbols = 0
            for i in range(1, 17):
                count_symbol, = f.read(1)
                allSymbols += count_symbol
                offsets[i] = allSymbols

            assert allSymbols <= 162 # Too many symbols in Huffman table
            for j in range(allSymbols):
                symb, = f.read(1)
                symbols[j] = symb

            htable = HuffmanTable(tuple(offsets), tuple(symbols), [0] * len(symbols), set=True)
            if ACTable:
                self.header["HuffmanACTables"][tableID] = htable
            else:
                self.header["HuffmanDCTables"][tableID] = htable

            # print(ACTable, tableID)
            # for j in range(16):
            #     print(j + 1, end=': ')
            #     for k in range(offsets[j], offsets[j + 1]):
            #         print(hex(symbols[k])[2:], end=' ')
            #     print()

            L -= 17 + allSymbols
        assert L == 0
        return

    def readStartOfScan(self, f):
        print("Reading Start of Scan")
        assert self.header["numComponents"] != 0 # SOS detected before SOF
        L = self._uint(f.read(2))
        numComponents, = f.read(1) # Maybe be diff of last numComponents for progressive JPEG
        for _ in range(numComponents):
            componentID, = f.read(1)
            if self.header["zeroBased"]:
                componentID += 1
            assert componentID <= numComponents
            DCTableID, ACTableID = self.split_byte(f.read(1))
            assert DCTableID <= 3
            assert ACTableID <= 3
            self.header["ColorComponents"][componentID - 1].huffmanDCTableID = DCTableID
            self.header["ColorComponents"][componentID - 1].huffmanACTableID = ACTableID
        (self.header["startOfSelection"], self.header["endOfSelection"]) = f.read(2)
        self.header["successiveApproximationHigh"], self.header["successiveApproximationLow"] = self.split_byte(f.read(1))
        assert L == 6 + 2 * numComponents
        return

    def readRestartInterval(self, f):
        print("Reading DRI Value")
        L = self._uint(f.read(2))
        assert L == 4
        self.header["restartInterval"] = self._uint(f.read(2))
        return

    def readAPPN(self, f):
        print("Read APPN Marker")
        L = self._uint(f.read(2))
        f.seek(L - 2, 1)
        return

    def readComment(self, f):
        print("Read COM Marker")
        L = self._uint(f.read(2))
        f.seek(L - 2, 1)
        return

    def readJPG(self):
        with open(self.file_path, "rb") as f:
            last, current = f.read(2)
            if last != 0xFF or current != SOI:
                # Not a valid JPEG file
                self.header["valid"] = False
                return

            last, current = f.read(2)
            while self.header["valid"]:
                # Test fermeture fichier ?
                if last != 0xFF:
                    print("Expect a marker")
                    self.header["valid"] = False
                    return

                if current == SOF0:
                    self.readStartOfFrame(f)
                elif current == DQT:
                    self.readQuantizationTable(f)
                elif current == DHT:
                    self.readHuffmanTable(f)
                elif current == SOS:
                    self.readStartOfScan(f)
                    # break from while loop after SOS
                    break
                elif current == DRI:
                    self.readRestartInterval(f)
                elif APP0 <= current <= APP15:
                    self.readAPPN(f)
                elif current == COM:
                    self.readComment(f)
                elif (
                    JPG0 <= current <= JPG13
                    or current == DNL
                    or current == DHP
                    or current == EXP
                ):
                    self.readComment(f)
                elif current == TEM:
                    pass  # TEM has no size
                elif current == 0xFF:
                    # any number of 0xFF in a row is allowed and should be ignored
                    current, = f.read(1)
                    continue
                elif current == SOI:
                    # Embedded JPGs not supported
                    self.header["valid"] = False
                    return
                elif current == EOI:
                    # EOI detected before SOS
                    self.header["valid"] = False
                    return
                elif current == DAC:
                    # Arithmetic Coding not supported
                    self.header["valid"] = False
                    return
                elif SOF0 <= current <= SOF15:
                    # SOF marker not supported 0x
                    self.header["valid"] = False
                    return
                elif RST0 <= current <= RST7:
                    # RSTN detected before SOS
                    self.header["valid"] = False
                    return
                else:
                    # Unkown marker -> print current
                    self.header["valid"] = False
                    return
                last, current = f.read(2)

            if self.header["valid"]:
                huffman_data = []
                current, = f.read(1)
                # read compressed image data
                while True:
                    last = current
                    current, = f.read(1)
                    if last == 0xFF:
                        # end of image
                        if current == EOI:
                            break
                        # 0xFF00 means put al literal 0xFF in image data and ignore 0x00
                        elif current == 0x00:
                            huffman_data.append(last)
                            current, = f.read(1)
                        elif RST0 <= current <= RST7:
                            # overwrite marker with next byte
                            current, = f.read(1)
                        elif current == 0xFF:
                            continue
                        else:
                            raise ValueError()
                    else:
                        huffman_data.append(last)
        self.header["huffmanData"] = huffman_data
        # Validate Huffman Header Info
        return self.header

    # generate all Huffman codes based on symbols from a Huffman table
    def generateCodes(self, htable):
        code = 0
        for i in range(16):
            for j in range(htable.offsets[i], htable.offsets[i + 1]):
                htable.codes[j] = code
                code += 1
            code <<= 1 # Append a zero to the right
        return

    # return the symbol from the Huffman table that corresponds to the next Huffman code read from the BitReader
    def getNextSymbol(self, b, htable):
        currentCode = 0
        for i in range(16):
            bit = int('0b' + b.read(1).bin, 2)
            # bit = int(b.read(1).bin)
            currentCode = (currentCode << 1) | bit
            for j in range(htable.offsets[i], htable.offsets[i + 1]):
                if currentCode == htable.codes[j]:
                    return htable.symbols[j]
        return -1 # Error

    def decodeMCUComponent(self, b, previousDC, dcTable, acTable):
        component = [0] * 64
        L = self.getNextSymbol(b, dcTable)
        assert L != 0xFF # Invalid DC value
        assert L <= 11 # DC coefficient length greater than 11

        B = b.read(L).bin
        coeff = int('0b' + B, 2) if B else 0

        assert coeff != -1
        if (L != 0 and coeff > (1 << (L - 1))):
            coeff -= (1 << L) - 1
            # coeff = - (coeff ^ (1 << L)) # inverse all bits
        component[0] = coeff + previousDC
        previousDC = component[0]

        i = 1
        while i < 64:
            symbol = self.getNextSymbol(b, acTable)
            assert symbol != 0xFF # Invalid AC value

            # 0x00 measn fill remainder of component with 0
            if symbol == 0x00:
                # Rempir le reste par des 0
                # Deja fait par defaut donc on ne fait rien
                return component, previousDC
            numZeros, coeffLengts = self.split_byte(struct.pack('B', symbol))
            # coeff = 0
            if symbol == 0xF0:
                numZeros = 16

            assert i + numZeros < 64 # Zero run-length exceeded MCU
            for j in range(numZeros):
                component[self.zigzagMap[i]] = 0
                i += 1 # Avant ou apres ??

            assert coeffLengts <= 10 # AC coefficient length greater than 10
            if coeffLengts != 0:
                coeff =  int('0b' + b.read(coeffLengts).bin, 2)
                assert coeff != -1 # Invalid AC value
                if (coeff > (1 << (coeffLengts - 1))):
                    coeff -= (1 << coeffLengts) - 1
                component[self.zigzagMap[i]] = coeff
                i += 1

        return component, previousDC

    # decode all the Huffman data and fill all MCUs
    def decodeHuffmanData(self):
        mcuHeight = (self.header["height"] + 7) // 8
        mcuWidth = (self.header["width"] + 7) // 8
        mcus = [[0 for _ in range(self.header[ 'numComponents' ])] for _ in range(mcuHeight * mcuWidth)]
        hdata = self.header['huffmanData']
        bits_data = [Bits(uint=n, length=8) for n in hdata]
        b = ConstBitStream(bits_data[0])
        for bit in bits_data[1:]:
            b += bit
        for dc_table in self.header["HuffmanDCTables"]:
            if dc_table:
                self.generateCodes(dc_table)
        for ac_table in self.header["HuffmanACTables"]:
            if ac_table:
                self.generateCodes(ac_table)

        # BitReader
        previousDCs = [0, 0, 0]
        restart_inteval = self.header["restartInterval"]
        for i in range(mcuHeight * mcuWidth):
            if restart_inteval and i % restart_inteval == 0:
                previousDCs = [0, 0, 0]
                b.align()
            for j in range(self.header["numComponents"]):
                # self.decodeMCUComponent(b, mcus[i][j])
                # Single Channel of a single mcu
                dc_table = self.header["HuffmanDCTables"][self.header["ColorComponents"][j].huffmanDCTableID]
                ac_table = self.header["HuffmanACTables"][self.header["ColorComponents"][j].huffmanACTableID]
                component, previousDC = self.decodeMCUComponent(b, previousDCs[j], dc_table, ac_table)
                previousDCs[j] = previousDC
                mcus[i][j] = component

        return mcus

    def writeBMP(self, mcus):
        height, width = self.header["height"], self.header["width"]
        file_name = self.file_path.stem
        bmp_name = file_name + ".bmp"
        mcuHeight = (height + 7) // 8
        mcuWidth = (width + 7) // 8
        paddingSize = width % 4
        size = 14 + 12 + height * width * 3 + height * paddingSize
        short_int = struct.Struct('<H')
        long_int = struct.Struct('<I')
        with open(bmp_name, "wb") as f:
            # BMP Header
            f.write(struct.pack('B', ord('B')))
            f.write(struct.pack('B', ord('M')))
            f.write(long_int.pack(size))
            f.write(long_int.pack(0))
            f.write(long_int.pack(0x1A))

            # DIB Header
            f.write(long_int.pack(12))
            f.write(short_int.pack(width))
            f.write(short_int.pack(height))
            f.write(short_int.pack(1))
            f.write(short_int.pack(24))

            for y in range(height): # TODO reverse(range(height))
                mcuRow, pixelRow = y // 8, y % 8
                for x in range(width):
                    mcuCol, pixelCol = x // 8, x % 8
                    mcuIndex = mcuRow * mcuWidth + mcuCol
                    pixelIndex = pixelRow * 8 + pixelCol
                    f.write(struct.pack('3B', mcus[mcuIndex][2][pixelIndex], mcus[mcuIndex][1][pixelIndex], mcus[mcuIndex][0][pixelIndex]))
                    # f.write(struct.pack('3B', mcus[mcuIndex][0][pixelIndex], mcus[mcuIndex][1][pixelIndex], mcus[mcuIndex][2][pixelIndex])) #TODO
                for i in range(paddingSize):
                    f.write(struct.pack('B', 0))


from pathlib import Path
jpeg_test = Path('/home/mratet/PycharmProjects/stanford_compression_library/scl/compressors/tests_jpeg')
test_file = 'cat.jpg'
jpeg_reader = JPEGReader(jpeg_test / test_file)
header = jpeg_reader.readJPG()
mcus = jpeg_reader.decodeHuffmanData()
jpeg_reader.writeBMP(mcus)
a = 1

# const byte zigZagMap[] = {
#     0,   1,  8, 16,  9,  2,  3, 10,
#     17, 24, 32, 25, 18, 11,  4,  5,
#     12, 19, 26, 33, 40, 48, 41, 34,
#     27, 20, 13,  6,  7, 14, 21, 28,
#     35, 42, 49, 56, 57, 50, 43, 36,
#     29, 22, 15, 23, 30, 37, 44, 51,
#     58, 59, 52, 45, 38, 31, 39, 46,
#     53, 60, 61, 54, 47, 55, 62, 63
# };


def convert_to_01(img):
    return (img / 255).astype(np.float32)


def convert_to_int(img):
    return (img * 255).astype(np.uint8)


def array_to_blocks(arr, block_size):
    """
    2D array [H,W] to [num_blocks, block_size, block_size]
    """
    H, W = arr.shape

    H = (H // block_size) * block_size
    W = (W // block_size) * block_size
    arr = arr[:H, :W]

    # Reshape and transpose to get blocks
    return (
        arr.reshape(H // block_size, block_size, -1, block_size)
        .swapaxes(1, 2)
        .reshape(-1, block_size, block_size)
    )


def blocks_to_array(blocks, H, W):
    """
    inverse of array_to_blocks
    """
    Bh, Bw = blocks.shape[1], blocks.shape[2]

    # Ensure that the total number of elements matches
    if blocks.size != H * W:
        raise ValueError(
            "Total number of elements in blocks does not match the expected array size"
        )

    # Reshape back to the original array
    return blocks.reshape(H // Bh, W // Bw, Bh, Bw).swapaxes(1, 2).reshape(H, W)


def format_block(block):
    if len(block.shape) < 3:
        return block.astype(float) - 128
    # [0, 255] -> [-128, 127]
    block_centered = block[:, :, 1].astype(float) - 128
    return block_centered


def invert_format_block(block):
    # [-128, 127] -> [0, 255]
    new_block = block + 128
    # in process of dct and inverse dct with quantization,
    # some values can go out of range
    new_block[new_block > 255] = 255
    new_block[new_block < 0] = 0
    return new_block


##### Color Treatment
# Correction de gamma ?
# Possible d'ecrire le tout avec des multiplications matricielles ?


def rgb2yuv(r, g, b):  # in (0,255) range
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.0877 * (r - y)
    return y, u, v


def yuv2rgb(y, u, v):
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y * 2.03211 * u
    return r, g, b


def rgb2ycbcr(r, g, b):  # in (0,255) range
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr2rgb(y, cb, cr):
    r = y + 1.4075 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.7790 * (cb - 128)

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    return r, g, b


### Color_subsampling
def chroma_subsampling(chroma):
    # 4:2:0
    # Subsampling selon H, W de dim 2. On garde le top-left, pas le mean
    blocks = array_to_blocks(chroma, 2)
    new_chroma = np.array([[[block[0, 0]]] for block in blocks])
    sub_chroma = blocks_to_array(new_chroma, 2, 2)
    return sub_chroma.repeat(2, axis=0).repeat(2, axis=1)


##### JPEG Encoder
### DCT
def get_DCT_matrix(L):
    C = np.zeros((L, L))
    for k in range(L):
        for n in range(L):
            if k == 0:
                C[k, n] = np.sqrt(1 / L)
            else:
                C[k, n] = np.sqrt(2 / L) * np.cos((np.pi * k * (0.5 + n)) / L)
    return C



def DCT_2D(img):
    H, W = img.shape
    C1 = get_DCT_matrix(H)
    C2 = get_DCT_matrix(W)
    return C1 @ img @ C2.T


def dct_2d(block):
    return fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")


def idct_2d(block):
    return fftpack.idct(fftpack.idct(block.T, norm="ortho").T, norm="ortho")


### Quantization
# Plusieurs quantization table selon la tache -> a voir pour l'integration


def get_quantization_table():
    quant_table = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )
    return quant_table


def get_chroma_quantization_table():
    return np.array(
        [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ]
    )


def get_80_quality_quantization_table():
    return np.array(
        [
            [6, 4, 4, 6, 10, 16, 20, 24],
            [5, 5, 6, 8, 10, 23, 24, 22],
            [6, 5, 6, 10, 16, 23, 28, 22],
            [6, 7, 9, 12, 20, 35, 32, 25],
            [7, 9, 15, 22, 27, 44, 41, 31],
            [10, 14, 22, 26, 32, 42, 45, 37],
            [20, 26, 31, 35, 41, 48, 48, 40],
            [29, 37, 38, 39, 45, 40, 41, 40],
        ]
    )


def quantize(block):
    quant_table = get_quantization_table()
    return (block / quant_table).round().astype(np.int32)


def dequantize(block):
    quant_table = get_quantization_table()
    return (block * quant_table).astype(np.float32)


### RLE Encoding / Huffman Encoding




# type(f) -> io.BufferedReader
# f.read(size) : On lit les `size` premier bytes
# f.tell()     : Position du curseur
# f.seek(offset, whence) : deplacement du curser a `offset` du point de ref. Si whence 0 = debut, 1 = pos_actuel, 2 = fin
# PREFIX = b"\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46" # A valid JPEG prefix

# (int).to_bytes(size, 'little')
# struct.pack / unpack
# Concretement, on definit un format dans struck qui nous permet ensuite de lire le buffer
# struct.unpack(format, buffer)
# struct.iter_unpack(format, buffer) : Besoin de verifier la taille du buffer, utile pour lire plusieurs chunks a la suite
# struct.calcsize(format) pour verifier notre format
# Voir cette page pour la creation de format : https://docs.python.org/3/library/struct.html#format-characters
# Possible d'utiliser un hex editor pour explorer les fichiers binaires -> HxD, hexdump
exit()


dataset = load_dataset("Freed-Wu/kodak")["test"]
sample_image = dataset[0]["image"]
# sample_image.show()
img = np.array(sample_image)
H, W, D = img.shape

r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
y, cb, cr = rgb2ycbcr(r, g, b)

cb = chroma_subsampling(cb)
cr = chroma_subsampling(cr)


BLOCK_SIZE = 8
blocks = array_to_blocks(y, BLOCK_SIZE)

print(blocks.shape)

for block in blocks:
    block_coeff = np.round(DCT_2D(block - 128), 3)
    quant_coeff = quantize(block_coeff)
    # Remplacement du coeff DC par DCi - DC(i-1)
    # Huffman table sont soit predefinies, soit calcule on-the-fly
    A = 1
