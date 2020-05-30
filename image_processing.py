{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# simple preprocessor class\n",
    "\n",
    "class SimplePreprocessor():\n",
    "    def __init__(self,width,heigth,inter=cv.INTER_AREA):\n",
    "        self.width = width\n",
    "        self.height = height\n",
    "        self.inter = inter\n",
    "        \n",
    "    def preprocessor(self,image):\n",
    "        return cv.resize(image,(self.width,self.heigth),\n",
    "                        interpolation = self.inter)\n",
    "        \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DataLoader():\n",
    "    def __init__(self,preprocessors=None):\n",
    "        self.preprocessors = preprocessor\n",
    "        \n",
    "        if self.preprocessors is None:\n",
    "            self.preprocessors = []\n",
    "    \n",
    "    def load(self,imagePaths,verbose=1):\n",
    "        data = []\n",
    "        labels = []\n",
    "        \n",
    "        for (i, imagePath) in enumerate (imagePaths):\n",
    "            image = cv.imread(imagePath)\n",
    "            label = imagePath.split(os.path.sep)[-2]\n",
    "            \n",
    "        if slef.preprocessors is not None:\n",
    "            for p in self.preprocessors:\n",
    "                image = p.preprocess(image)\n",
    "        \n",
    "        data.append(image)\n",
    "        labels.append(label)\n",
    "        \n",
    "        \n",
    "        if verbose > 0 and i > 0 and (i + 1) % verbose == 0: \n",
    "            print(\"[INFO] processed {}/{}\".format(i + 1,\n",
    "               len(imagePaths)))\n",
    "            \n",
    "        # return a tuple of the data and labels\n",
    "        return (np.array(data), np.array(labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
