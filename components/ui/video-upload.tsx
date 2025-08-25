// components/ui/video-upload.tsx
import { useState, ChangeEvent } from 'react';

interface Props {
  onUploadSuccess: () => void;
}

export function VideoUpload({ onUploadSuccess }: Props) {
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        onUploadSuccess();
      } else {
        console.error('Upload failed');
      }
    } catch (error) {
      console.error('Error during upload:', error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg">
      <h2 className="text-xl font-bold mb-4">1. Upload Video</h2>
      <input type="file" accept="video/mp4" onChange={handleFileChange} disabled={isUploading} 
        className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-500 file:text-white hover:file:bg-indigo-600"/>
      {isUploading && <p className="mt-4 text-indigo-400">Uploading...</p>}
    </div>
  );
}