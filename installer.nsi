# This installs two files, app.exe and logo.ico, creates a start menu shortcut, builds an uninstaller, and
# adds uninstall information to the registry for Add/Remove Programs
 
# Setup a Modern Interface
!include "MUI.nsh"
Unicode True

# Set LZMA Compression
!ifdef NSIS_LZMA_COMPRESS_WHOLE
SetCompressor lzma
!else
SetCompressor /SOLID lzma
!endif

# Variables For Later Use, related with the App
!define APPNAME "Scatmm"
!define COMPANYNAME "CEMOP"
!define DESCRIPTION "Graphical interface for the Scattering Matrix Method (similar to the Transfer Matrix Method)"
# These three must be integers
!define VERSIONMAJOR 3
!define VERSIONMINOR 8
!define VERSIONBUILD 2
# These will be displayed by the "Click here for support information" link in "Add/Remove Programs"
# It is possible to use "mailto:" links in here to open the email client
!define HELPURL "mailto:m.alexandre@campus.fct.unl.pt" # "Support Information" link
!define UPDATE "https://github.com/perspe/scatmm"
 
RequestExecutionLevel admin ;Require admin rights on NT6+ (When UAC is turned on)

# Default Path to install the App
InstallDir "$PROGRAMFILES\${COMPANYNAME}\${APPNAME}"

# This will be in the installer/uninstaller's title bar
Name "${COMPANYNAME} - ${APPNAME}"
!define MUI_ICON "logo.ico"
!define MUI_UNICON "logo.ico"
outFile "dist\${APPNAME}_v${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}.exe"
 
# Just three pages - license agreement, install location, and installation
!define MUI_PAGE_HEADER_TEXT "${APPNAME}"
!define MUI_PAGE_HEADER_SUBTEXT "${DESCRIPTION}"
!define MUI_LICENSEPAGE_TEXT_TOP "${APPNAME}"
!define MUI_LICENSEPAGE_BUTTON "I Agree"
!define MUI_FINISHPAGE_TEXT_LARGE
!define MUI_FINISH_PAGE_TEXT "${APPNAME} has been installed successfully.\nClick Finish to close this wizard."
# Add the severall instalation pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENCE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"


!include LogicLib.nsh

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin" ;Require admin rights on NT4+
        messageBox mb_iconstop "Administrator rights required!"
        setErrorLevel 740 ;ERROR_ELEVATION_REQUIRED
        quit
${EndIf}
!macroend
 
function .onInit
	setShellVarContext all
	!insertmacro VerifyUserIsAdmin
functionEnd
 
section "install" INSTALL
	# Files for the install directory - to build the installer, these should be in the same directory as the install script (this file)
	setOutPath "$INSTDIR"
	# Files added here should be removed by the uninstaller (see section "uninstall")
	File /r /x *.so dist\${APPNAME}\*
	# File /r dist\SpectoInterp\* /oname=PATH -- to output to a path different from setOutPath
	# Add any other files for the install directory (license files, app data, etc) here
	# Uninstaller - See function un.onInit and section "uninstall" for configuration
	writeUninstaller "$INSTDIR\uninstall.exe"
	# Start Menu
	createDirectory "$SMPROGRAMS\${COMPANYNAME}"
	createShortCut "$SMPROGRAMS\${COMPANYNAME}\${APPNAME}.lnk" "$INSTDIR\${APPNAME}.exe" "" "$INSTDIR\_internal\logo.ico"
	createShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\${APPNAME}.exe" "" "$INSTDIR\_internal\logo.ico"
	# Registry information for add/remove programs
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "DisplayName" "${COMPANYNAME} - ${APPNAME}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "QuietUninstallString" "$\"$INSTDIR\uninstall.exe$\" /S"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "InstallLocation" "$\"$INSTDIR$\""
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "DisplayIcon" "$\"$INSTDIR\_internal\logo.ico$\""
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "Publisher" "${COMPANYNAME}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "HelpLink" "${HELPURL}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "URLUpdateInfo" "${UPDATE}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "URLInfoAbout" "${UPDATE}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "VersionMinor" ${VERSIONMINOR}
	# There is no option for modifying or repairing the install
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "NoModify" 1
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}" "NoRepair" 1
sectionEnd

# Uninstaller
function un.onInit
	SetShellVarContext all
	#Verify the uninstaller - last chance to back out
	MessageBox MB_OKCANCEL "Permanantly remove ${APPNAME}?" IDOK next
		Abort
	next:
	!insertmacro VerifyUserIsAdmin
functionEnd
 
section "uninstall"
	# Remove Start Menu launcher
	Delete "$SMPROGRAMS\${COMPANYNAME}\${APPNAME}.lnk"
	# Try to remove the Start Menu folder - this will only happen if it is empty
	RMDir "$SMPROGRAMS\${COMPANYNAME}"
	# Always delete uninstaller as the last action
	Delete $INSTDIR\uninstall.exe
	RMDir /r $INSTDIR
	# Remove uninstaller information from the registry
	DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${APPNAME}"
sectionEnd