%% Part 1 - 2.1: Getting Started
addpath matlab_files;
Imagestrain = loadImagesInDirectory ('training-set/23x28/');
[Imagestest, Identity] = loadTestImagesInDirectory ('testing-set/23x28/');
% There are 40 unique faces in the test directory in total (Note this!)

% Additional code: Build histogram to show that; number of bins=40
% Will be used at Part 2.9.
[testLabelCounts, ~] = histcounts(Identity, 1:max(Identity)+1);

%% Part 2.2 - 2.3: Compute of the mean, eigVals and EigFaces in Facespace:
Means = floor(mean(Imagestrain));
CenteredVectors = (Imagestrain - repmat(Means, size(Imagestrain,1), 1));
CovarianceMatrix = cov(CenteredVectors);

[U, S, V] = svd(CenteredVectors);
Space = V(: ,1:size(Imagestrain,1))';
Eigenvalues = diag(S);

%% Part 2.4: Display of the mean image:
MeanImage = uint8 (zeros(28, 23));
for k = 0:643
    MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1);
end
figure; imshow(MeanImage); title('Mean Image');
clear k; % Workspace clear-up -- DEBUG

%% Part 2.5: Display of the 20 first eigenfaces: (Coded by me)
figure;
for i = 1:20
    subplot(4,5,i);
    %  This notation reqd to have show images, otherwise images are B&W.
    imshow(reshape(Space(i,:),[28,23]),[]);
end
sgtitle("First 20 Eigenfaces");
clear i; % workspace clean-up -- DEBUG

%% Part 2.6:  Projection of the two sets of images onto the face space:
Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);

%% Part 2.7: Compute dist from projected test to proj'd training images:
Threshold = 20; % Number of eigenfaces to be used.

%Distances contains for each test image, the distance to every train image
Distances=zeros(size(Locationstest,1), size(Locationstrain,1));

for i=1:size(Locationstest,1)
    for j=1:size(Locationstrain,1)
        Sum=0;
        for k=1: Threshold
            Sum=Sum+(Locationstrain(j,k)-Locationstest(i,k)).^2;
        end
        Distances(i,j)=Sum;
    end
end

% Sort best matching training images for test images (using min distance):
Values = zeros(size(Locationstest,1), size(Locationstrain,1));
Indices = zeros(size(Locationstest,1), size(Locationstrain,1));
for i=1:size(Locationstest,1)
    [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end

clear i;
clear Sum; clear j; clear k; % workspace clean-up -- DEBUG

%% Part 2.8: Display of first 6 recognition results, image per image:
figure; x=6; y=2; % x,y subplot parameters.
for i=1:6
    Image = uint8 (zeros(28, 23));
    for k = 0:643
        Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
    end
    subplot (x,y,2*i-1);
    imshow (Image);
    title('Image tested');
    
    Imagerec = uint8 (zeros(28, 23));
    for k = 0:643
        Imagerec( mod (k,28)+1, floor(k/28)+1 ) = ...
            Imagestrain ((Indices(i,1)),k+1);
    end
    subplot (x,y,2*i);
    imshow (Imagerec);
    title(['Image recognised with ', num2str(Threshold), ...
        ' eigenfaces:',num2str((Indices(i,1)))]);
end
clear i; clear k; clear x; clear y; % workspace clean-up -- DEBUG

%% Part 2.9: Compute recognition rate with 20 eigenfaces:
% Part 2.10 uses this section. Code can be modified and migrated here.
recognised_person = zeros(1,40); % 40 different faces in testing images
recognitionrate=zeros(1,5);  
number_per_number=zeros(1,5);

% Loop over the testing images:
test_idx = 1;
while test_idx < size(Imagestest,1)
    id = Identity(test_idx); % Returns the label for the image, 1:40
    
    % Extract minimum value and the corresponding index:
    % Note that you extract only the closest index (KNN = 1)
    distmin = Values(id,1); indicemin = Indices(id,1);
    
    % If the test samples deplete, or the id extracted in the outer and 
    % inner loop (Identity(test_idx)) do not match break from inner loop, 
    % then update the predictions. Identity is sorted.
    while (test_idx < 70) && (Identity(test_idx) == id)
        % Find the minimum distance and indices (KNN = 1)
        if (Values(test_idx,1) < distmin)
            distmin=Values(test_idx,1);
            indicemin=Indices(test_idx,1);
        end
        % id updates slower than the test samples. Normal as there are 40
        % faces available but 70 samples.
        test_idx = test_idx+1;
    end
    
    % Log the recognized face (Best matching training image):
    recognised_person(id) = indicemin;
    
    % TestLabelCounts coming from Part 2.1 histogram. Tells how many 
    % samples per each face we have. Same face is in test set 0 to 5 times
    % number_per_number counts how many times a id is seen (1, 2 ,3, 4, 5
    % times) in the test dataset. Will be used to normalize recognition
    % rate. The second while loop passes through the same "id" 
    % testLabelCounts(id) times. Bins this and averages later.
    number_per_number(testLabelCounts(id)) = ... 
        number_per_number(testLabelCounts(id))+1; 
    
    % Does a transformation on the indexing. Checks whether the labeling is 
    % correct by comparing transformed training label to testing label.
    % (200/40 = 5 samples per face)
    if (id==floor((indicemin-1)/5)+1) % If correct person is recognized
        recognitionrate(testLabelCounts(id))= ...
            recognitionrate(testLabelCounts(id))+1; 
        % Increment the recognition, given 
    end 
end

% Calculates recognition rate, 
for  i=1:5
    recognitionrate(i)=recognitionrate(i)/number_per_number(i);
end
mean_recognition_rate = mean(recognitionrate);

%% Part 2.10: Different number of eigenfaces' affect on recognition rate:
averageRR=zeros(1,20);
for t=1:40
    % Repeats after Part 2.7: Compute Distances
    Threshold = t;
    Distances=zeros(size(Locationstest,1), size(Locationstrain,1));
    for i = 1:size(Locationstest,1)
        for j = 1: size(Locationstrain,1)
            Sum = 0;
            for k = 1:Threshold
                Sum = Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
            end
            Distances(i,j)=Sum;
        end
    end
    
    % Repeats after Part 2.7: Sort distances
    Values=zeros(size(Locationstest,1), size(Locationstrain,1));
    Indices=zeros(size(Locationstest,1), size(Locationstrain,1));
    number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here
    for i=1:70
        number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
        [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
    end
    
    % Repeats after Part 2.9:
    recognised_person=zeros(1,40);
    recognitionrate=zeros(1,5);
    number_per_number=zeros(1,5);
    
    i=1;
    while (i<70)
        id=Identity(1,i);
        distmin=Values(id,1);
        indicemin=Indices(id,1);
        while (i<70)&&(Identity(1,i)==id)
            if (Values(i,1)<distmin)
                distmin=Values(i,1);
                indicemin=Indices(i,1);
            end
            i=i+1;
        end
        
        recognised_person(1,id)=indicemin;
        number_per_number(number_of_test_images(1,id))= ...
            number_per_number(number_of_test_images(1,id))+1;
        if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
            recognitionrate(number_of_test_images(1,id))= ...
                recognitionrate(number_of_test_images(1,id))+1;
        end
    end
    
    % Calculates recognition rate, logs it.
    for  i=1:5
        recognitionrate(1,i)=recognitionrate(1,i)/number_per_number(1,i);
    end
    averageRR(1,t)=mean(recognitionrate(1,:));
end
figure;
plot(averageRR);
title('Recognition rate against the number of eigenfaces used');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 2.11: Investigate the effect of K in KNN classifier:
% effect of K: You need to evaluate the effect of K in KNN and
% plot the recognition rate against K. Use 20 eigenfaces here.

% Set to 20 Eigenfaces: Part 2.6 and 2.7
Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);
Threshold = 20; % Number of eigenfaces to be used.
Distances=zeros(size(Locationstest,1), size(Locationstrain,1));
for i=1:size(Locationstest,1)
    for j=1:size(Locationstrain,1)
        Sum=0;
        for k=1: Threshold
            Sum=Sum+(Locationstrain(j,k)-Locationstest(i,k)).^2;
        end
        Distances(i,j)=Sum;
    end
end
% Sort best matching training images for test images (using min distance):
Values = zeros(size(Locationstest,1), size(Locationstrain,1));
Indices = zeros(size(Locationstest,1), size(Locationstrain,1));
for i=1:size(Locationstest,1)
    [Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end

recog_rate=zeros(1,20);
% Redo the experiment at Part 2.9 for 20 Eigenfaces and for K from 1 to 20.
for K = 1:size(Imagestrain,1)
    % Indices (70x200) hold values sorted from min to max for each row.
    % i.e, for each test, best fitting training samples are ordered with
    % increasing distance. (x-1)/5+1 is transformation to extract training
    % labels (used in previous sections)
    index_matrix = floor((Indices(:,:)-1)/5)+1;
    
    % For each test sample check the recognition
    correct_count = 0; % initialize correct count for each K tested.
    for idx = 1:size(Imagestest,1)
        id = Identity(idx); % Returns the label for the image, 1:40
        
        % Fetch K closest indices, use mode to find which class is assigned
        % If equal mode, smallest one is selected by MATLAB (shortcoming?)
        knn_output = mode(index_matrix(idx, 1:K));
        
        if (id == knn_output) % The recognition is correct
            correct_count = correct_count + 1;
        end    
    end
    
    % Traversed through all samples, compute recognition rate:
    recog_rate(K) = correct_count ./ size(Imagestest,1); 
    
end

% Plot the Recognition Rate:
figure;
plot(recog_rate);
title('Recognition rate against the number of NNs (with 20 Eigenfaces)');
ylabel('Recognition Rate'); xlabel('K');
