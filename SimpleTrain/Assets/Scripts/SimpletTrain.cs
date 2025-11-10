using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
public class SimpletTrain : Agent
{

    public Rigidbody target;
    Rigidbody rb;
    float timer = 0f;
    float lastDistance = 0f;
    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }


    public override void OnEpisodeBegin()
    {
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        rb.position = new Vector3(Random.Range(-4.5f, 4.5f), 0.5f, Random.Range(-4.5f, 4.5f));
        target.position = new Vector3(Random.Range(-4.5f, 4.5f), 1f, Random.Range(-4.5f, 4.5f));
        timer = 0f;
        lastDistance = Vector3.Distance(rb.position, target.position);

    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(target.position - rb.position);
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.z);
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        timer += Time.deltaTime;


        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        Vector3 force = new Vector3(moveX, 0, moveZ) * 100f;
        rb.AddForce(force);

        float distanceToTarget = Vector3.Distance(rb.position, target.position);
        float distanceReward = lastDistance - distanceToTarget;

        AddReward(distanceReward * 0.1f);
        lastDistance = distanceToTarget;

        if (distanceToTarget < 1.5f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (rb.position.y < 0 || timer > 20f)
        {
            EndEpisode();
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
