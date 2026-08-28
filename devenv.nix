{ pkgs, ... }:

{
  # Reproducible dev shell for this learning repo.
  # Enter with: devenv shell
  # Optional direnv integration: direnv allow

  packages = [
    pkgs.zig
    pkgs.jujutsu
    pkgs.git
    pkgs.gh
  ];

  scripts.zig-test.exec = ''
    zig build test
  '';

  scripts.zig-run.exec = ''
    zig build run
  '';

  enterShell = ''
    echo "zigging-ai-models dev shell"
    zig version
    jj --version
  '';

  enterTest = ''
    zig version
    jj --version
    git --version
    zig build test
  '';
}
