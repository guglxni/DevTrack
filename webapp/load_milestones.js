const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

/**
 * Loads milestone data from CSV files in the data directory
 * @returns {Promise<Array>} - A promise that resolves to an array of milestone objects
 */
function loadMilestones() {
  return new Promise((resolve, reject) => {
    const dataDir = path.join(__dirname, '..', 'data');
    const csvFiles = fs.readdirSync(dataDir)
      .filter(file => file.endsWith('.csv') && file.includes('CDDC') && file.includes('Table 1'));
    
    if (csvFiles.length === 0) {
      reject(new Error('No CSV files found in data directory'));
      return;
    }
    
    console.log(`Found ${csvFiles.length} CSV files with milestone data`);
    
    const milestones = [];
    let filesProcessed = 0;
    
    csvFiles.forEach(csvFile => {
      const filePath = path.join(dataDir, csvFile);
      const domainMatch = csvFile.match(/CDDC\s+(\w+)-Table/);
      const domain = domainMatch ? domainMatch[1] : 'Unknown';
      
      console.log(`Processing domain: ${domain} from file: ${csvFile}`);
      
      // Track the current age range for rows with empty age
      let currentAgeRange = '';
      
      fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (data) => {
          // Extract data from the CSV
          // First, identify the column names which might vary by file
          const ageRangeKey = Object.keys(data).find(key => key.includes('Age'));
          const behaviorKey = Object.keys(data).find(key => 
            key !== ageRangeKey && 
            !key.includes('Criteria') && 
            !key.includes('SI.No') && 
            !key.includes('train')
          );
          const criteriaKey = Object.keys(data).find(key => key.includes('Criteria'));
          
          if (!behaviorKey) return; // Skip if missing behavior column
          
          // Get the age range, using the current one if this row's is empty
          const rowAgeRange = data[ageRangeKey]?.replace('months', '').replace('m', '').trim();
          if (rowAgeRange && rowAgeRange !== 'Age') {
            currentAgeRange = rowAgeRange;
          }
          
          const behavior = data[behaviorKey]?.trim();
          const criteria = data[criteriaKey]?.trim() || '';
          
          // Skip rows with empty behaviors or if age range is a header row
          if (!behavior || behavior === 'Checklist' || !currentAgeRange || currentAgeRange === 'SI.No') return;
          
          // Create milestone object
          const milestone = {
            behavior: behavior,
            domain: domain,
            age_range: currentAgeRange,
            criteria: criteria
          };
          
          milestones.push(milestone);
          console.log(`Added milestone: ${milestone.behavior} (${milestone.domain}, ${milestone.age_range})`);
        })
        .on('end', () => {
          filesProcessed++;
          if (filesProcessed === csvFiles.length) {
            console.log(`Successfully loaded ${milestones.length} milestones from ${csvFiles.length} CSV files`);
            resolve(milestones);
          }
        })
        .on('error', (err) => {
          console.error(`Error processing ${csvFile}: ${err.message}`);
          reject(err);
        });
    });
  });
}

module.exports = { loadMilestones }; 