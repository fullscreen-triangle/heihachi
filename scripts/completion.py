#!/usr/bin/env python3
"""
Shell completion script generator for Heihachi CLI.

This script generates shell completion functions for bash, zsh, and fish shells
to enable command-line completion for the Heihachi CLI.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cli.enhanced_cli import create_parser

def generate_bash_completion() -> str:
    """Generate bash completion script.
    
    Returns:
        Bash completion script content
    """
    parser = create_parser()
    
    # Extract argument information
    arguments = []
    for action in parser._actions:
        if action.option_strings:  # Only include optional arguments
            arguments.extend(action.option_strings)
    
    # Build completion script
    script = """
# Bash completion script for Heihachi audio analysis framework
# Place this file in /etc/bash_completion.d/ or source it from your .bashrc

_heihachi()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Define all options
    opts="ARGUMENTS"
    
    # Handle specific argument completions
    case "${prev}" in
        -c|--config)
            # Complete with yaml files
            COMPREPLY=( $(compgen -f -X '!*.@(yaml|yml)' -- "${cur}") )
            return 0
            ;;
        -o|--output|--export-dir)
            # Complete with directories
            COMPREPLY=( $(compgen -d -- "${cur}") )
            return 0
            ;;
        --export)
            # Complete with export formats
            local formats="json csv yaml md markdown html xml"
            COMPREPLY=( $(compgen -W "${formats}" -- "${cur}") )
            return 0
            ;;
        --configs)
            # Complete with comma-separated yaml files
            if [[ "${cur}" == *,* ]]; then
                local prefix=${cur%,*},
                local last=${cur##*,}
                COMPREPLY=( $(compgen -f -X '!*.@(yaml|yml)' -P "${prefix}," -- "${last}") )
            else
                COMPREPLY=( $(compgen -f -X '!*.@(yaml|yml)' -- "${cur}") )
            fi
            return 0
            ;;
        *)
            ;;
    esac
    
    # Default completion for arguments
    if [[ ${cur} == -* ]] ; then
        # Complete option flags
        COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
        return 0
    else
        # Complete with files and directories
        COMPREPLY=( $(compgen -f -- "${cur}") $(compgen -d -- "${cur}") )
        return 0
    fi
}

complete -F _heihachi heihachi
"""
    
    # Substitute arguments into script
    script = script.replace("ARGUMENTS", " ".join(arguments))
    
    return script

def generate_zsh_completion() -> str:
    """Generate zsh completion script.
    
    Returns:
        Zsh completion script content
    """
    parser = create_parser()
    
    # Extract argument information
    arguments = []
    help_text = {}
    for action in parser._actions:
        if action.option_strings:  # Only include optional arguments
            arg = action.option_strings[-1]  # Use the longest option string
            arguments.append(arg)
            help_text[arg] = action.help or ''
    
    # Build completion script
    script = """
#compdef heihachi

# Zsh completion script for Heihachi audio analysis framework
# Place this file in a directory in your $fpath or use with compinit

_heihachi() {
    local -a commands
    local -a options
    
    options=(
ARGUMENT_DEFINITIONS
    )
    
    _arguments -C \\
        '1:input file or directory:_files' \\
        '*:: :->args' \\
        && return 0
    
    case "$state" in
        args)
            _describe -t options "options" options && return 0
            ;;
    esac
    
    return 1
}

_heihachi_config() {
    local -a config_files
    config_files=( $(find . -name "*.yaml" -o -name "*.yml" 2>/dev/null) )
    compadd -a config_files
}

_heihachi_export_formats() {
    local -a formats
    formats=(
        'json:JavaScript Object Notation'
        'csv:Comma-separated values'
        'yaml:YAML Ain't Markup Language'
        'md:Markdown format'
        'markdown:Markdown format'
        'html:HTML format'
        'xml:XML format'
    )
    _describe 'export format' formats
}

_heihachi "$@"
"""
    
    # Build argument definitions
    arg_definitions = []
    for arg in arguments:
        help_str = help_text.get(arg, '').replace("'", "").replace(':', '')
        
        # Handle special cases for completions
        if arg in ('--config', '-c'):
            arg_definitions.append(f"    '{arg}[{help_str}]:config file:_heihachi_config'")
        elif arg in ('--output', '-o', '--export-dir'):
            arg_definitions.append(f"    '{arg}[{help_str}]:directory:_files -/'")
        elif arg == '--export':
            arg_definitions.append(f"    '{arg}[{help_str}]:export format:_heihachi_export_formats'")
        elif arg == '--configs':
            arg_definitions.append(f"    '{arg}[{help_str}]:config files:_heihachi_config'")
        else:
            arg_definitions.append(f"    '{arg}[{help_str}]'")
    
    # Substitute argument definitions into script
    script = script.replace("ARGUMENT_DEFINITIONS", "\n".join(arg_definitions))
    
    return script

def generate_fish_completion() -> str:
    """Generate fish completion script.
    
    Returns:
        Fish completion script content
    """
    parser = create_parser()
    
    # Extract argument information
    arguments = []
    help_text = {}
    for action in parser._actions:
        if action.option_strings:  # Only include optional arguments
            for arg in action.option_strings:
                arguments.append(arg)
                help_text[arg] = action.help or ''
    
    # Build completion script
    script = """
# Fish completion script for Heihachi audio analysis framework
# Place this file in ~/.config/fish/completions/heihachi.fish

function __fish_heihachi_no_subcommand
    set -l cmd (commandline -poc)
    if [ (count $cmd) -eq 1 ]
        return 0
    end
    return 1
end

# Complete the input file/directory
complete -c heihachi -n "__fish_heihachi_no_subcommand" -a "(__fish_complete_path)"

# Add command options
ARGUMENT_COMPLETIONS

# Special completions for specific options
complete -c heihachi -n "__fish_seen_subcommand_from -c; or __fish_seen_subcommand_from --config" -a "(find . -name '*.yaml' -o -name '*.yml' | string replace './' '')" -d "Config file"

complete -c heihachi -n "__fish_seen_subcommand_from --export" -a "json" -d "JavaScript Object Notation"
complete -c heihachi -n "__fish_seen_subcommand_from --export" -a "csv" -d "Comma-separated values"
complete -c heihachi -n "__fish_seen_subcommand_from --export" -a "yaml" -d "YAML format"
complete -c heihachi -n "__fish_seen_subcommand_from --export" -a "md" -d "Markdown format"
complete -c heihachi -n "__fish_seen_subcommand_from --export" -a "html" -d "HTML format"
complete -c heihachi -n "__fish_seen_subcommand_from --export" -a "xml" -d "XML format"

complete -c heihachi -n "__fish_seen_subcommand_from -o; or __fish_seen_subcommand_from --output; or __fish_seen_subcommand_from --export-dir" -a "(__fish_complete_directories)" -d "Directory"
"""
    
    # Build argument completions
    arg_completions = []
    for arg in arguments:
        help_str = help_text.get(arg, '').replace("'", "").replace('"', '')
        arg_completions.append(f'complete -c heihachi -n "__fish_use_subcommand" -l "{arg.lstrip("-")}" -d "{help_str}"')
    
    # Substitute argument completions into script
    script = script.replace("ARGUMENT_COMPLETIONS", "\n".join(arg_completions))
    
    return script

def main():
    """Main function to generate and save completion scripts."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate shell completion scripts for Heihachi CLI')
    parser.add_argument('--output-dir', '-o', default='./completions',
                       help='Directory to save completion scripts')
    parser.add_argument('--bash', action='store_true', default=True,
                       help='Generate bash completion script')
    parser.add_argument('--zsh', action='store_true', default=True,
                       help='Generate zsh completion script')
    parser.add_argument('--fish', action='store_true', default=True,
                       help='Generate fish completion script')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate and save completion scripts
    if args.bash:
        bash_script = generate_bash_completion()
        with open(output_dir / 'heihachi-completion.bash', 'w') as f:
            f.write(bash_script)
        print(f"Bash completion script saved to {output_dir}/heihachi-completion.bash")
        print("To use, add this to your .bashrc:")
        print(f"  source {output_dir.resolve()}/heihachi-completion.bash")
    
    if args.zsh:
        zsh_script = generate_zsh_completion()
        with open(output_dir / '_heihachi', 'w') as f:
            f.write(zsh_script)
        print(f"Zsh completion script saved to {output_dir}/_heihachi")
        print("To use, add this directory to your fpath in .zshrc:")
        print(f"  fpath=({output_dir.resolve()} $fpath)")
        print("  autoload -Uz compinit && compinit")
    
    if args.fish:
        fish_script = generate_fish_completion()
        with open(output_dir / 'heihachi.fish', 'w') as f:
            f.write(fish_script)
        print(f"Fish completion script saved to {output_dir}/heihachi.fish")
        print("To use, copy or link this file to ~/.config/fish/completions/:")
        print(f"  ln -s {output_dir.resolve()}/heihachi.fish ~/.config/fish/completions/")

if __name__ == '__main__':
    main() 