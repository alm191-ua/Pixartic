Convierte fácilmente tus imágenes en pixel art. Es posible seleccionar resoluciones y paletas de color personalizadas.\
Además, se pueden añadir fácilmente nuevas paletas.

## Cómo usarlo
```
usage: pixartic.py [-h] [-p] [-r RESOLUTION RESOLUTION]
                   [-P {ANSI_FORLORN_64,AROACIDIC,AZURE_ABYSS,A_BIZARRE_CONCOTION,EXOPHOBIA,FISTAT6,LOSPEC500,MIDNIGHT_ABLAZ,PAITO24,PAPER10,PIXLS_DEFAULT,P_1BIT_MONITOR_GLOW,ROSE_BUS,SHAG_CARPET}]
                   [-s]
                   image

positional arguments:
  image                 path to the image

optional arguments:
  -h, --help            show this help message and exit
  -p, --proportions     keep proportions
  -r RESOLUTION RESOLUTION, --resolution RESOLUTION RESOLUTION
                        resolution of the final image
  -P {ANSI_FORLORN_64,AROACIDIC,AZURE_ABYSS,A_BIZARRE_CONCOTION,EXOPHOBIA,FISTAT6,LOSPEC500,MIDNIGHT_ABLAZ,PAITO24,PAPER10,PIXLS_DEFAULT,P_1BIT_MONITOR_GLOW,ROSE_BUS,SHAG_CARPET}, --pallette {ANSI_FORLORN_64,AROACIDIC,AZURE_ABYSS,A_BIZARRE_CONCOTION,EXOPHOBIA,FISTAT6,LOSPEC500,MIDNIGHT_ABLAZ,PAITO24,PAPER10,PIXLS_DEFAULT,P_1BIT_MONITOR_GLOW,ROSE_BUS,SHAG_CARPET}
                        color pallette to use
  -s, --shadowing       use shadowing (do not work)

Example:
  python pixartic.py -p -r 64 64 -P AZURE_ABYSS imgs/myImage.png
```

## Incluir nuevas paletas
Es posible incluir nuevas paletas directamente en el fichero [pallettes.py](./pallettes/pallettes.py), así como en un nuevo fichero e importarlo desde [pallettes.py](./pallettes/pallettes.py).

También es posible utilizar la herramienta [getRGB.py](./getRGB.py) para convertir ficheros de paletas en hexadecimal a código python e incluir las paletas en el código de forma automática.
```
usage: getRGB.py [-h] mode filename

Convert .hex file to .txt file with RGB values

positional arguments:
  mode        the mode to run the script in (use 'import' for import the pallettes automatically, other for do not import)
  filename    the .hex file to convert

optional arguments:
  -h, --help  show this help message and exit

Example:
  python getRGB.py import myPallette.hex

```

> NOTA: es posible obtener ficheros .hex con paletas de color en la web [lospec.com](https://lospec.com/palette-list).
