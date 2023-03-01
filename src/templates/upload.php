
    <?php
    $filename = $_FILE['file']['name'];
    $location = 'upload/'.$filename;
    if(move_uploaded_file($_FILES['file']['tmp_name'],$location)){
        echo'FILE UPLOADED SUCCESSFULLY'
        header('Location: /path/to/homepage.html');
    }
    else{
        echo'ERROR UPLOADING FILE';
    }
    ?>
