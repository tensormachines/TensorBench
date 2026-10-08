# LJM Software Installer

To install:

```
  $ sudo ./labjack_ljm_installer.run
```

 - installs .so library files to `/usr/local/lib`
 - installs .h header files to `/usr/local/include`
 - installs LJM/Kipling files to `/usr/local/share`
 - sets up LabJack device rule files
 - sets up LabJack file permissions
 - installs Kipling to `/opt/labjack_kipling`
 - installs a Kipling command-line launcher `/usr/local/bin/labjack_kipling`


## Makeself

labjack_ljm_installer.run is a [Makeself script](http://megastep.org/makeself/).
