; What if no faces are detected !?! ...

FileCount = %1%

WorkingDirectory = picasa_process ; Name of the working directory.

PicasaPath = C:\Program Files (x86)\Google\Picasa3\Picasa3.exe
PicasaID = Picasa 3

; Coordinates of some point in the green album (Albums tab)
AlbumX = 40
AlbumY = 115

; Coordinates of the center of the star that appears while Picasa is processing
; the images for face detection
StarXWithoutAlbum = 41
StarXWithAlbum = 41
StarYWithoutAlbum = 118
StarYWithAlbum = 162

AlbumColor = 0xB3F4C9
StarColor = 0xFEFEFE
BGColor = 0xD8B5A2 ; Color behind the center of the star

; Coordinates of some middle point on the People tab
PeopleXWithoutAlbum = 135
PeopleXWithAlbum = 135
PeopleYWithoutAlbum = 130
PeopleYWithAlbum = 175

; Coordinates of some middle point on the Expand Groups button (People section)
ExpandGroupsX = 1085
ExpandGroupsY = 115

; Coordinates of some "safe" point in the area where the faces are displayed
; (People section)
FacesX = 1085
FacesY = 175

; Coordinates of some middle point on the Export button (People section)
ExportX = 750
ExportY = 960

WinClose, %WorkingDirectory% ; The purpose of this will become clear below.
Run, %PicasaPath%, , max
WinWait, %PicasaID%
WinActivate, %PicasaID%

; Detecting if the Albums tab is displayed or not.
Sleep, 3500
PixelSearch, x, y, %AlbumX%, %AlbumY%, %AlbumX%, %AlbumY%, %AlbumColor%, 0
if ErrorLevel = 0
{
	StarX = %StarXWithAlbum%
	StarY = %StarYWithAlbum%
	PeopleX = %PeopleXWithAlbum%
	PeopleY = %PeopleYWithAlbum%
	;MsgBox, With Album
}
else
{
	StarX = %StarXWithoutAlbum%
	StarY = %StarYWithoutAlbum%
	PeopleX = %PeopleXWithoutAlbum%
	PeopleY = %PeopleYWithoutAlbum%
	;MsgBox, Without Album
}

; Detecting the beginning of the face detection process
StartTime := A_TickCount
Loop
{
	if (FileCount < 10 && A_TickCount - StartTime > 10000)
		break
	PixelSearch, x, y, %StarX%, %StarY%, %StarX%, %StarY%, %StarColor%, 0
	if ErrorLevel = 0
		break
}

;MsgBox, Face detection has begun

; Detecting the end of the face detection process
Loop
{
	PixelSearch, x, y, %StarX%, %StarY%, %StarX%, %StarY%, %BGColor%, 0
	if ErrorLevel = 0
		break
}

; Exporting results
Click %PeopleX%, %PeopleY%
Click %ExpandGroupsX%, %ExpandGroupsY%
Click %FacesX%, %FacesY%
Click %ExportX%, %ExportY%
Send Faces
Send {Enter}

; Waiting until the export is done. (Picasa will open the WorkingDirectory
; folder with a new 'Faces' folder in it)
WinWait, %WorkingDirectory%

WinClose, %PicasaID%