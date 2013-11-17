import win32security
import win32file

DIR = 'Test'
USER = 'Utilisateurs'

info = win32security.DACL_SECURITY_INFORMATION
sd = win32security.GetFileSecurity(DIR, info)
acl = sd.GetSecurityDescriptorDacl()
sidUser = win32security.LookupAccountName(None, USER)[0]
acl.AddAccessAllowedAce(win32file.FILE_ALL_ACCESS, sidUser)
sd.SetSecurityDescriptorDacl(1, acl, 0)
win32security.SetFileSecurity(DIR, info, sd)