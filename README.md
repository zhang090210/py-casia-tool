# py-casia-tool

> ⚠️ **WARNING: UNDER DEVELOPMENT**
> 
> This project is still under active development and is not ready for production use. APIs may change without notice, and functionality may be incomplete or broken.

A Python tool for decoding CASIA (Chinese Academy of Sciences Institute of Automation) offline handwritten Chinese character datasets.

## Features

- Decode `.gnt` (offline isolated character) files
- Decode `.dgrl` (offline handwritten text line) files
- Unified decode API for multiple formats
- Returns structured data with numpy arrays for bitmaps

## Installation

```bash
# Install with pip
pip install py-casia-tool

# Or install from source
git clone https://github.com/Zzy/py-casia-tool.git
cd py-casia-tool
uv install
```

## Usage

### Decode from file path

```python
from py_casia_tool import decode

# Auto-detect format by file suffix
samples = decode("path/to/file.gnt")
samples = decode("path/to/file.dgrl")

# Specify format explicitly
samples = decode("path/to/file", fmt="gnt")
```

### Decode from bytes

```python
from py_casia_tool import decode

with open("file.gnt", "rb") as f:
    data = f.read()

samples = decode(data, fmt="gnt")
```

### Decode with writer code parsing

```python
from py_casia_tool import decode

# For files named like "1246-f.gnt", extract writer code "1246"
samples = decode("1246-f.gnt", parse_writer_code=True)
for sample in samples:
    print(sample.writer_code)  # "1246"
```

### Direct decoder functions

```python
from py_casia_tool.decoders.gnt import decode_gnt
from py_casia_tool.decoders.dgrl import decode_dgrl

samples = decode_gnt("path/to/file.gnt")
lines = decode_dgrl("path/to/file.dgrl")
```

## API

### decode(source, fmt=None, parse_writer_code=False)

Unified decode function.

- `source`: File path (str/Path) or bytes
- `fmt`: Format name ("gnt" or "dgrl"), required when source is bytes
- `parse_writer_code`: Extract writer code from filename

### GntChar

Dataclass representing a single character from `.gnt` files:

| Field | Type | Description |
|-------|------|-------------|
| `tag_code` | str | GBK encoded character |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `bitmap` | np.ndarray | 2D numpy array of shape (height, width) |
| `writer_code` | Optional[str] | Writer identifier |

### DgrlLine

Dataclass representing a text line from `.dgrl` files:

| Field | Type | Description |
|-------|------|-------------|
| `line_number` | int | Line number |
| `image_width` | int | Full image width |
| `image_height` | int | Full image height |
| `label` | str | Text label |
| `width` | int | Character width |
| `height` | int | Character height |
| `top` | int | Top position |
| `left` | int | Left position |
| `bitmap` | np.ndarray | 2D numpy array |

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=py_casia_tool

# Run specific tests
uv run pytest tests/test_gnt.py -v
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
