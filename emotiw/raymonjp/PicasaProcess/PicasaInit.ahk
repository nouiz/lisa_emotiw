PicasaPath = C:\Program Files (x86)\Google\Picasa3\Picasa3.exe
PicasaID = Picasa 3

WinClose, %PicasaID% ; Just in case.
Run, %PicasaPath%, , max
WinWait, %PicasaID%
WinActivate, %PicasaID%
Sleep, 1000
WinClose, %PicasaID%
