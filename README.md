# seam carving
Generates content aware resized gifs

## Example

<img src="example.gif" alt="Example gif" width="200"/>

## CLI

```Animated seam carving 
Authors: DasEtwas, Friz64

USAGE:
    seam-carving [OPTIONS] --input <FILE>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -f, --fps <FPS>            GIF FPS, is converted into GIF units (10ms) (float) [default: 20]
    -i, --input <FILE>         Input file path
    -l, --length <SECONDS>     GIF length in seconds (float) [default: 1.0]
    -o, --output <FILE>        Output file [default: output.gif]
    -q, --quality <QUALITY>    GIF Quality [1 - 100] [default: 30]
    -s, --scale <PERCENT>      Final scale of the last frame in percent (float) [default: 10]
```
