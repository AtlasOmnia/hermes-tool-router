# Patch File Removed From Public Folder

The generated patch was intentionally removed from the public package because it was produced against a local baseline that contained a private hardcoded persona. Even though the patch removed that persona, deleted diff lines would still expose private material.

Use `hermes-agent-files/` as the public source overlay/reference instead, or regenerate a patch from a clean public upstream baseline that never contained private persona text.
