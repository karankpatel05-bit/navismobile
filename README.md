# Navis Mobile

Navis Mobile is a mobile robot AI assistant project.

## Project Setup

### Option 1: Visual Studio Code (Windows)
1. **Prerequisites Checklist**: Ensure you have downloaded and installed **Git for Windows** and **Visual Studio Code**.
2. **Open your Folder**: Launch VS Code, click `File` > `Open Folder`, and select the `navismobile` directory.
3. **The Source Control Tab**: On the far-left sidebar, click the **Source Control** icon (it looks like a branch node, shortcut `Ctrl + Shift + G`).
4. **Staging (Adding) Files**: When you make changes to files and save them, they will appear under the "Changes" list in this tab. Click the **`+` (plus)** icon that appears when you hover over a file. This is the equivalent of `git add`.
5. **Committing**: Above the staged files, there is a "Message" text box. Type what you changed (e.g., "Updated firmware logic") and click the **Commit** button.
6. **Syncing/Pushing**: After committing, a **Sync Changes** button will appear in the panel. Click it to push your commits to GitHub. Since VS Code uses an integrated Git Credential Manager, the very first time you click "Sync Changes", a browser window will automatically pop up, allowing you to log into GitHub safely with a single click (no need for manual Personal Access Tokens).
7. **Pulling**: If you want to download updates from GitHub, click the `...` menu at the top of the Source Control tab and select **Pull**.

### Option 2: Command Line (Ubuntu/Terminal)
1. **Navigate to Project**: Open your terminal (`Ctrl+Alt+T`) and move to your project folder:
   ```bash
   cd /path/to/navis_mobile_robot/
   ```
2. **Check Status**: It's good practice to run `git status`. It will tell you exactly which files have been modified. 
3. **Stage Your Changes**: Add all modified or newly created files by running:
   ```bash
   git add .
   ```
4. **Create the Commit**: Save a point in the project's history with a detailed message:
   ```bash
   git commit -m "Added tracking feature to YOLO"
   ```
5. **Pushing Changes**: Upload your newly created commits to the remote `.git` repository on GitHub:
   ```bash
   git push origin main
   ```
6. **Pulling Changes**: If you've pushed updates from your Windows computer and want them to appear on your Ubuntu machine, run:
   ```bash
   git pull origin main
   ```

## How to Use "Navis Mobile" Step-by-Step
1. **Hardware Setup**: Flash your Arduino/Raspberry Pi Pico W using the `pico_firmware.ino` script via the Arduino IDE! Connect it to your computer or mobile.
2. **Install Python and pip**: Ensure Python 3.8+ is installed.
3. **Create a Virtual Environment**:
   * Windows: `python -m venv venv` and `venv\Scripts\activate`
   * Ubuntu: `python3 -m venv venv` and `source venv/bin/activate`
4. **Install Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
5. **Environment Variables**: Rename `.env.example` to `.env` and fill in the necessary API keys.
6. **Start Application**: Run the Python server:
   ```bash
   python app.py
   ```
7. Use the interface as directed by the web app.
