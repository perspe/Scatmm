ECHO Running pyinstaller.....

pyinstaller --clean --noconfirm Scatmm.spec

ECHO NSIS Script

makensis installer.nsi

ECHO Finished....
