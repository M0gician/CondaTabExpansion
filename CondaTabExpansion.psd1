@{

    # Script module or binary module file associated with this manifest.
    RootModule = 'CondaTabExpansion.psm1'
    
    # Version number of this module.
    ModuleVersion = '0.1.1'
    
    # ID used to uniquely identify this module
    GUID = '325e4a50-8f4b-40c5-857e-8755c2415c58'
    
    # Author of this module
    Author = 'm0gician Yang'
    
    # Description of the functionality provided by this module
    Description = 'Provides samrt tab completion for conda commands, parameters, remotes and branch names.'

    # List of all files packaged with this module
    FileList = @('CondaTabExpansion.ps1', 'CondaTabExpansion.psm1')
    
    # Minimum version of the Windows PowerShell engine required by this module
    PowerShellVersion = '3.0'
    
    # Functions to export from this module
    FunctionsToExport = '*'
    
    # Cmdlets to export from this module
    CmdletsToExport = @()
    
    # Variables to export from this module
    VariablesToExport = @()
    
    # Aliases to export from this module
    AliasesToExport = '*'
    
    # Private data to pass to the module specified in RootModule/ModuleToProcess.
    # This may also contain a PSData hashtable with additional module metadata used by PowerShell.
    PrivateData = @{
    
        PSData = @{
            # Tags applied to this module. These help with module discovery in online galleries.
            Tags = @('conda', 'anaconda', 'prompt', 'tab', 'tab-completion', 'tab-expansion', 'tabexpansion')
    
            # A URL to the main website for this project.
            ProjectUri = 'https://github.com/M0gician/Conda-Tab-Expansion'
        }
    
    }
    
}