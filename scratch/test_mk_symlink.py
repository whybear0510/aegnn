import click
import pathlib

def create_folder_with_symlink(folder_name, symlink_name):
    folder_path = pathlib.Path(folder_name)
    symlink_path = pathlib.Path(symlink_name)

    # Check if the folder already exists
    if folder_path.exists():
        click.confirm(f"Folder '{folder_name}' already exists. Continue?", default=True, abort=True)

    # Create the folder
    folder_path.mkdir(parents=True, exist_ok=True)

    # If the symlink file already exists, remove it
    if symlink_path.exists():
        symlink_path.unlink()

    # Create the symlink
    symlink_path.symlink_to(folder_path)

    click.echo(f"Folder '{folder_path.name}' created with '{symlink_path.name}' -> '{folder_path.name}'")

if __name__ == '__main__':
    create_folder_with_symlink(folder_name = '/users/yyang22/thesis/aegnn_project/aegnn/scratch/a', symlink_name = './a_ln_s')
