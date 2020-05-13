if (Get-Module condaProfile) { return }

$thisDirectory = "$env:USERPROFILE\Documents\PowerShell\Scripts\Conda-Tab-Expansion"
. $thisDirectory\CondaTabExpansion.ps1

Export-ModuleMember -Alias refreshenv -Function 'TabExpansion'