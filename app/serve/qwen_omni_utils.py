def process_mm_info(conversation, use_audio_in_video=False):
    """
    Process multimodal information from conversation format.
    This function extracts audio, image, and video data from a conversation format
    used by the Qwen models.
    
    Args:
        conversation: A list of conversation turns with multimodal content
        use_audio_in_video: Whether to include audio in video processing
    
    Returns:
        tuple: (audios, images, videos) - Processed multimodal data
    """
    audios = []
    images = []
    videos = []
    
    for turn in conversation:
        if 'content' in turn:
            for item in turn['content']:
                if item.get('type') == 'audio':
                    audio_path = item.get('audio')
                    if audio_path:
                        audios.append(audio_path)
                elif item.get('type') == 'image':
                    image_path = item.get('image')
                    if image_path:
                        images.append(image_path)
                elif item.get('type') == 'video':
                    video_path = item.get('video')
                    if video_path:
                        videos.append(video_path)
    
    # This is a simplified version - in a real implementation you would:
    # - Load the actual audio/image/video files
    # - Process them appropriately for the model
    # - Return tensors or other processed formats
    
    # For now, just return the paths as-is if they exist, otherwise None
    audios = audios if audios else None
    images = images if images else None
    videos = videos if videos else None
    
    return audios, images, videos