This directory contains the source files for AEPsych's Docusaurus website demos.

See the website's [README](../website/README.md) for additional information on how to build and start the website locally.

# Adding New Demos

To add a new demo to your project, follow these steps:

1. Create a new Markdown file in the `demos/markdown/` directory. Choose a descriptive name for the file that reflects the demo content. For example, you can create a file named `ParticleEffectDemo.md`.

2. Open the Markdown file and structure it as follows:

```markdown
   ## Particle Effect Demo

   This text will be rendered above the video.

   <video controls muted style="width: 100%;">
     <source src="VIDEO_URL_HERE" type="video/mp4">
     Your browser does not support the video tag.
   </video>

   This text will be rendered below the video.
```

Replace VIDEO_URL_HERE with the URL of the video asset. You can host the video on GitHub by following the steps below:

- Convert the video files to the .mp4 format if they are not already in that format.
- Host the videos as assets using GitHub Issues:
    1) Drag and drop the video files into the Issues tab of your GitHub repository.
    2) Wait for the video to finish loading.
    3) Once the video is fully loaded, it will provide an asset URL. You don't need to hit the "submit" button.

**Note**: Video must be less than 100MB.

## Adding .zip file assets
1) Add the necessary zip files for both Mac and Windows versions of the demo. Place the zip files in the `demos/` directory to ensure they are accessible for the demo parsing function. Follow the naming convention `<name>_Win.zip` and `<name>_Mac.zip`.

Your file structure should look like this:
```
demos-|
      markdown-|
               |- ParticleEffectDemo.md
      |- ParticleEffectDemo_Mac.zip
      |- ParticleEffectDemo_Win.zip
```
**Note**: If the size of the demo file is larger than 25MB, it is recommended to push the file to Git Large File Storage (LFS).

By following this structured approach, you create an organized repository that facilitates efficient access to the demo files.

Employing Git LFS to track large files is essential for maintaining a lightweight repository and preventing large binary files from inflating the version control history.

**Docs**: [Git LFS Documentation](https://git-lfs.com/)

### Git LFS Workflow:

To seamlessly integrate the zip files while utilizing Git LFS, adhere to the following workflow:

1. **Open Terminal and Navigate to Repository**:
Open your terminal and navigate to the root directory of your AEPsych repository.

2. **Initialize Git LFS**:
Use the following command to initialize Git LFS in your repository:
```sh
git lfs install
```
### Configure `.gitattributes`:

Add the file name to the `.gitattributes` file located in the root directory. Utilize the following syntax to specify that the file should be tracked using Git LFS:

```sh
demos/ParticleEffectDemo.md filter=lfs diff=lfs merge=lfs -text
```

### Check Staging Status:
View the status of items staged for commit using:
```sh
git lfs status
```
### Track the File:
Employ the following command to begin tracking the demo file using Git LFS:
```sh
git lfs track demos/ParticleEffectDemo.md
```
### List Tracked Files:
To observe all files being tracked by Git LFS, run:
```sh
git lfs ls-files
```
### Stage and Commit Changes to git:
```sh
git add .
git commit -m "Add large files using Git LFS"
```
## Update the website Navigation

2) Update the `demo.json` file located in `website/demo.json`. This file contains the configuration for all demos in the `demos/` directory. The title here is used to render the title of the demo on the demoSidebar component.

Add a new demo object to represent the newly added demo:
```json
{
  "id": "ParticleEffectDemo",
  "title": "Particle Effect Demo"
}
```

Ensure that the "id" matches the name of the Markdown file (without the file extension) and corresponds to the names of the zip files.


### Modifying Demo Text
To add or modify the text associated with a specific demo, follow these steps:

Locate the Markdown file corresponding to the demo you wish to modify, e.g., `demos/markdown/ParticleEffectDemo.md`.

Open the Markdown file in a text editor and modify the text as desired. You can place the text before or after the video element.

#### Adding Demo to the Overview Page
To add the demo to the overview page, follow these steps:

Go to `website/pages/demos/index.js`.

Add a DemoButton component for the demo:

Example:
```js
<DemoButton
  imageUrl={`${this.props.config.baseUrl}img/particle-effect-demo.png`}
  demoUrl="/demos/ParticleEffectDemo"
  buttonText="Particle Effect Demo"
/>
```
Ensure that you have the associated image file in the `website/static/img/` directory. In this example, the image file should be named `particle-effect-demo.png`.

#### Updating Demo Photo
To update the photo associated with a specific demo, follow these steps:

1) Locate the current photo file for the demo, e.g., `website/static/img/particle-effect-demo.png`.
2) Replace the existing photo file with the new photo file you want to use. Make sure the new photo file has the same name (`particle-effect-demo.png`) and is in the same location (`website/static/img/`).
