if (Get-Module CondaTabExpansion) { return }

. "$PSScriptRoot\CondaTabExpansion.ps1"

Export-ModuleMember -Alias refreshenv -Function 'TabExpansion'